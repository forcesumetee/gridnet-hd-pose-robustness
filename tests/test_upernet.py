import json
import os

import numpy as np
import pytest
torch = pytest.importorskip("torch")   # GPU-side tests: skipped on the torch-less Mac venv
import torch.nn as nn
from PIL import Image

from gridnet.train.data2d import CropDataset, class_weights_2d, list_pairs
from gridnet.train.engine2d import Trainer2D, seg_loss_2d
from gridnet.train.upernet import UPerNet, sliding_window, window_starts


class DummyEncoder(nn.Module):
    """Four feature levels at strides 4/8/16/32 (stands in for a timm backbone in tests)."""

    def __init__(self, chs=(8, 16, 24, 32)):
        super().__init__()
        self.chs = chs
        self.s4 = nn.Conv2d(3, chs[0], 4, stride=4)
        self.s8 = nn.Conv2d(chs[0], chs[1], 2, stride=2)
        self.s16 = nn.Conv2d(chs[1], chs[2], 2, stride=2)
        self.s32 = nn.Conv2d(chs[2], chs[3], 2, stride=2)

    def forward(self, x):
        f4 = self.s4(x); f8 = self.s8(f4); f16 = self.s16(f8); f32 = self.s32(f16)
        return [f4, f8, f16, f32]


def tiny_model():
    enc = DummyEncoder()
    return UPerNet(encoder=enc, encoder_channels=enc.chs, num_classes=11, channels=16, feat_dim=8, pretrained=False)


def test_forward_shapes_and_feature_head():
    m = tiny_model().eval()
    x = torch.randn(2, 3, 64, 96)
    logits, feat = m(x, return_feat=True)
    assert logits.shape == (2, 11, 16, 24) and feat.shape == (2, 8, 16, 24)


def test_window_starts_cover_the_image_and_end_flush():
    assert window_starts(100, 40, 30) == [0, 30, 60]
    assert window_starts(64, 64, 32) == [0]
    assert window_starts(50, 64, 32) == [0]


def test_sliding_window_matches_direct_forward_without_overlap():
    m = tiny_model().eval()
    img = torch.randn(3, 64, 128)
    lg, ft = sliding_window(m, img, crop=64, stride=64, batch=2, amp=False)
    assert lg.shape == (11, 16, 32) and ft.shape == (8, 16, 32)
    for x0 in (0, 64):                                # each window equals its own direct forward
        d, df = m(img[None, :, :, x0:x0 + 64], return_feat=True)
        assert torch.allclose(lg[:, :, x0 // 4:x0 // 4 + 16], d[0], atol=1e-4)
        assert torch.allclose(ft[:, :, x0 // 4:x0 // 4 + 16], df[0], atol=1e-4)
    lg2, _ = sliding_window(m, torch.randn(3, 72, 100), crop=64, stride=32, batch=3, amp=False)
    assert lg2.shape == (11, 18, 25)                 # odd sizes: padded, stitched, cropped back


def make_zone(root, zone, n=3, size=(160, 120), seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, zone, "images")); os.makedirs(os.path.join(root, zone, "masks"))
    for k in range(n):
        img = rng.integers(0, 255, (size[1], size[0], 3), np.uint8)
        Image.fromarray(img).save(os.path.join(root, zone, "images", f"img{k}.JPG"), quality=90)
        m = rng.choice([5, 14, 16, 19, 255], size=(size[1], size[0]), p=[0.05, 0.3, 0.5, 0.1, 0.05]).astype(np.uint8)
        m[10:20, 30:60] = 1                           # original id 1 = cat-head pylon -> class 0 (rare)
        Image.fromarray(m).save(os.path.join(root, zone, "masks", f"img{k}.png"))


def test_crop_dataset_and_class_weights(tmp_path):
    root = str(tmp_path); make_zone(root, "zA", n=3); make_zone(root, "zB", n=2, seed=1)
    pairs = list_pairs(root, ["zA", "zB"])
    assert len(pairs) == 5 and list_pairs(root, ["zA"], per_zone=2) and len(list_pairs(root, ["zA"], per_zone=2)) == 2
    w = class_weights_2d(root, pairs, n=5, stride=2)
    assert w.shape == (11,) and w[0] > w[6]            # rare pylon gets a larger weight than herbaceous
    ds = CropDataset(root, pairs, crop=64, train=True, scales=(0.5, 1.0), rare_prob=1.0)
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and y.shape == (64, 64) and y.dtype == torch.long
    assert set(y.unique().tolist()) <= {0, 1, 4, 6, 8, 255}  # remapped to the 11-class space
    val = CropDataset(root, pairs, crop=64, train=False, seed=3, crops_per_image=2)
    a, b = val[1], val[1]
    assert len(val) == 10 and torch.equal(a[1], b[1])   # fixed crops are deterministic


def test_trainer_fit_eval_and_export_on_cpu(tmp_path):
    root = str(tmp_path / "data"); make_zone(root, "zA", n=3); make_zone(root, "zB", n=2, seed=1)
    pairs = list_pairs(root, ["zA"])
    train_ds = CropDataset(root, pairs, crop=64, train=True, scales=(1.0,))
    val_ds = CropDataset(root, list_pairs(root, ["zB"]), crop=64, train=False, crops_per_image=1)
    cfg = {"iters": 3, "batch": 2, "lr": 1e-3, "workers": 0, "amp": False, "log_every": 1, "ckpt_every": 2,
           "eval_every": 3, "lovasz": 1.0}
    tr = Trainer2D(tiny_model(), train_ds, val_ds, cfg, str(tmp_path / "ckpt"), class_w=np.ones(11, np.float32), device="cpu")
    assert tr.fit() == "done" and tr.it == 3
    assert os.path.exists(tmp_path / "ckpt" / "last.pth") and os.path.exists(tmp_path / "ckpt" / "best.pth")
    # resume into a fresh trainer and export the cache for one zone
    tr2 = Trainer2D(tiny_model(), None, None, cfg, str(tmp_path / "ckpt"), device="cpu")
    assert tr2.resume("best.pth") and tr2.it == 3
    n = tr2.export_cache(root, ["zB"], str(tmp_path / "cache2d"), crop=64, stride=64, batch=2, feat_pool=2, compact=False)
    assert n == 2
    z = np.load(tmp_path / "cache2d" / "zB" / "img0.npz")
    assert z["prob"].shape == (11, 30, 40) and z["prob"].dtype == np.float16     # 120x160 image -> 1/4
    assert z["feat"].shape == (8, 15, 20) and z["feat"].dtype == np.float16        # -> 1/8
    assert np.allclose(z["prob"].astype(np.float32).sum(0), 1.0, atol=2e-2)
    assert tr2.export_cache(root, ["zB"], str(tmp_path / "cache2d"), crop=64, stride=64) == 0  # skips existing
    # compact export (default): uint8 prob + int8 feat with per-channel scale; both formats read back the same
    from gridnet.train.maps2d import cached_image_ids, load_maps, load_prob_fp16
    assert tr2.export_cache(root, ["zB"], str(tmp_path / "compact"), crop=64, stride=64, batch=2, feat_pool=2) == 2
    c = np.load(tmp_path / "compact" / "zB" / "img0.npz")
    assert c["prob"].dtype == np.uint8 and c["feat"].dtype == np.int8 and c["feat_scale"].shape == (8,)
    assert np.abs(c["prob"].astype(np.float32) / 255 - z["prob"].astype(np.float32)).max() < 3e-3
    feat_c = c["feat"].astype(np.float32) * c["feat_scale"][:, None, None]
    assert np.abs(feat_c - z["feat"].astype(np.float32)).max() < 0.01 * np.abs(z["feat"].astype(np.float32)).max() + 1e-3
    a, b = load_maps(str(tmp_path / "cache2d"), "zB", "img0"), load_maps(str(tmp_path / "compact"), "zB", "img0")
    assert a["prob"].dtype == np.uint8 and np.abs(a["prob"].astype(int) - b["prob"].astype(int)).max() <= 1
    assert np.abs(a["feat"].astype(np.float32) * a["feat_scale"][:, None, None] - feat_c).max() < 0.02 * np.abs(feat_c).max() + 1e-3
    assert np.abs(load_prob_fp16(str(tmp_path / "compact"), "zB", "img0").astype(np.float32) - z["prob"].astype(np.float32)).max() < 3e-3
    assert cached_image_ids(str(tmp_path / "compact"), "zB") == ["img0", "img1"] and cached_image_ids(str(tmp_path), "nope") == []


def test_seg_loss_ignores_255_and_is_finite():
    logits = torch.randn(2, 11, 8, 8, requires_grad=True)
    target = torch.randint(0, 11, (2, 8, 8)); target[0, :4] = 255
    loss = seg_loss_2d(logits, target, None, lovasz_weight=1.0, lovasz_pixels=50)
    assert torch.isfinite(loss); loss.backward(); assert torch.isfinite(logits.grad).all()
