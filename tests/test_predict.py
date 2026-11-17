"""Predictions for zones without labels (leaderboard test zones) and their export, with a dummy backbone on the CPU."""
import json
import os
import sys

import numpy as np
import pytest
torch = pytest.importorskip("torch")   # GPU-side tests: skipped on the torch-less Mac venv
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fusion import DummySeg, write_zone  # noqa: E402

from gridnet import submission as S  # noqa: E402
from gridnet.train.data import load_zone  # noqa: E402
from gridnet.train.engine_fusion import FrozenBackbone, FusionTrainer, load_zone_context  # noqa: E402
from gridnet.train.fusion import ConcatHead, LateFusionHead  # noqa: E402
from gridnet.train.predict import head_from_checkpoint, predict_lidar, zone_is_labeled  # noqa: E402


def load_script(name):
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class NoisySeg(DummySeg):
    """A backbone whose output depends on torch's random stream, like LitePT's random serialization orders."""

    class _Backbone(DummySeg._Backbone):
        def forward(self, batch):
            out = super().forward(batch)
            out.feat = out.feat + 0.5 * torch.randn(out.feat.shape)
            return out

    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = self._Backbone(); self.head = nn.Linear(72, 11)


CFG = {"head": "late", "train_zones": [], "passes": 1, "its_per_zone": 1, "seed": 0, "radius": 25.0, "eval_max_points": 4000,
       "grid": 0.05, "lr": 1e-3, "head_chunk": 500}


def add_test_twin(cache, cache2d, data, src="zz", dst="tt", seed=3):
    """Copy zone `src` as an unlabeled zone `dst` (labels 255, labeled=False) and give both caches the raw-point
    index: every voxel has two raw points, in shuffled order. Returns {zone: (inverse, keep)}."""
    import shutil
    rng = np.random.default_rng(seed)
    out = {}
    for z in (src, dst):
        if z != src:
            for root in (cache, cache2d, data):
                shutil.copytree(os.path.join(root, src), os.path.join(root, z))
        with np.load(os.path.join(cache, src, "points.npz")) as p:
            arrays = {k: p[k] for k in p.files}
        n = len(arrays["labels"])
        inverse = rng.permutation(np.repeat(np.arange(n), 2)).astype(np.int32)
        keep = np.zeros(n, np.int32); keep[inverse[::-1]] = np.arange(2 * n - 1, -1, -1)      # first raw point of each voxel
        arrays.update(inverse=inverse, keep=keep, labeled=np.array(z == src))
        if z != src:
            arrays["labels"] = np.full(n, 255, np.uint8)
        np.savez(os.path.join(cache, z, "points.npz"), **arrays)
        out[z] = (inverse, keep)
    return out


def test_lidar_only_prediction_equals_the_second_output_of_the_fused_path(tmp_path):
    cache, cache2d, data = write_zone(str(tmp_path))
    torch.manual_seed(0)
    backbone = FrozenBackbone("", "", model=NoisySeg(), device="cpu")
    tr = FusionTrainer(backbone, LateFusionHead(), dict(CFG), str(tmp_path / "w"), device="cpu")
    ctx = load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), "zz", k=3, with_feat=False, device="cpu", verbose=False)
    fused, lidar = tr.predict_zone(ctx)
    alone = predict_lidar(backbone, load_zone(cache, "zz"), dict(CFG), "cpu")
    assert alone.dtype == np.uint8 and np.array_equal(alone, lidar) and not np.array_equal(alone, fused)
    torch.manual_seed(7); a = torch.rand(3)
    torch.manual_seed(7); predict_lidar(backbone, ctx.zone, dict(CFG), "cpu"); b = torch.rand(3)     # the outer stream is untouched
    assert torch.equal(a, b)
    assert zone_is_labeled(ctx.zone.labels) and not zone_is_labeled(np.full(5, 255, np.uint8)) and not zone_is_labeled(ctx.zone.labels, False)


def test_head_from_checkpoint_rebuilds_the_design(tmp_path):
    head = ConcatHead(lidar_input="prob")
    path = str(tmp_path / "best.pth")
    torch.save({"head": head.state_dict(), "cfg": {"head": "concat", "lidar_input": "prob", "trainer": "bank"}, "it": 4800, "best": 81.5}, path)
    h2, cfg, info = head_from_checkpoint(path)
    assert type(h2) is ConcatHead and cfg["lidar_input"] == "prob" and info == {"it": 4800, "best": 81.5}
    assert all(torch.equal(a, b) for a, b in zip(head.state_dict().values(), h2.state_dict().values()))


def test_predict_zones_script_and_export(tmp_path, monkeypatch, capsys):
    cache, cache2d, data = write_zone(str(tmp_path), n=6000)
    index = add_test_twin(cache, cache2d, data)
    torch.manual_seed(1)
    backbone = FrozenBackbone("", "", model=NoisySeg(), device="cpu")
    head = LateFusionHead()
    head_path = str(tmp_path / "late_a" / "best.pth"); os.makedirs(os.path.dirname(head_path))
    torch.save({"head": head.state_dict(), "cfg": {"head": "late", "trainer": "bank"}, "it": 4800, "best": 82.07}, head_path)
    mod = load_script("predict_zones")
    monkeypatch.setattr(mod, "FrozenBackbone", lambda *a, **k: backbone)
    out = str(tmp_path / "pred")
    argv = ["predict_zones.py", "--cache", cache, "--cache2d", cache2d, "--data", data, "--assoc-dir", str(tmp_path / "assoc"),
            "--litept", "x", "--backbone-ckpt", "y", "--head", head_path, "--out", out, "--zones", "zz,tt",
            "--eval-max-points", "4000", "--views", "3", "--head-chunk", "500", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv); mod.main()
    log = capsys.readouterr().out
    assert "head late from" in log and "(no labels)" in log and "fused mIoU" in log and "2/2 zones" in log
    rz, rt = (json.load(open(os.path.join(out, z, "predict.json"))) for z in ("zz", "tt"))
    fz, lz = np.load(os.path.join(out, "zz", "fused_pred.npy")), np.load(os.path.join(out, "zz", "lidar_pred.npy"))
    ft, lt = np.load(os.path.join(out, "tt", "fused_pred.npy")), np.load(os.path.join(out, "tt", "lidar_pred.npy"))
    assert fz.shape == (6000,) and fz.dtype == np.uint8 and np.array_equal(fz, ft) and np.array_equal(lz, lt)   # labels play no part
    assert rz["labeled"] and not rt["labeled"] and "metrics" in rz["fused"] and "metrics" not in rt["fused"]
    assert rt["voxels"] == 6000 and rt["images"] == 4 and rt["head_it"] == 4800 and sum(rt["lidar"]["counts"]) == 6000
    assert sorted(os.listdir(os.path.join(out, "tt"))) == ["fused_pred.npy", "lidar_pred.npy", "predict.json"]
    # the scores of the labelled zone are those of the validation code (FusionTrainer.evaluate) on the same zone
    tr = FusionTrainer(backbone, head, dict(CFG), str(tmp_path / "w"), device="cpu")
    loader = lambda zone, need_depths: load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), zone, k=3, with_feat=False,
                                                          device="cpu", verbose=False)
    ev = tr.evaluate(loader, ["zz"], save_dir=str(tmp_path / "w" / "eval"))
    assert abs(rz["fused"]["metrics"]["mIoU_present"] - ev["per_zone"]["zz"]["mIoU"]) < 1e-9
    assert abs(rz["lidar"]["metrics"]["mIoU_present"] - ev["per_zone"]["zz"]["lidar_only_mIoU"]) < 1e-9
    assert np.array_equal(np.load(str(tmp_path / "w" / "eval" / "zz" / "fused_pred.npy")), fz)
    capsys.readouterr()
    # second run: nothing is predicted again; --force does
    monkeypatch.setattr(sys, "argv", argv); mod.main()
    assert capsys.readouterr().out.count("predictions exist (skipped)") == 2
    before = os.path.getmtime(os.path.join(out, "tt", "predict.json"))
    monkeypatch.setattr(sys, "argv", argv + ["--force"]); mod.main()
    assert os.path.getmtime(os.path.join(out, "tt", "predict.json")) >= before and "(no labels)" in capsys.readouterr().out
    # LiDAR-only mode needs neither images nor a head and gives the same LiDAR prediction
    out2 = str(tmp_path / "pred_lidar")
    monkeypatch.setattr(sys, "argv", ["predict_zones.py", "--cache", cache, "--litept", "x", "--backbone-ckpt", "y", "--out", out2,
                                      "--zones", "tt", "--eval-max-points", "4000", "--device", "cpu"])
    mod.main()
    assert sorted(os.listdir(os.path.join(out2, "tt"))) == ["lidar_pred.npy", "predict.json"]
    assert np.array_equal(np.load(os.path.join(out2, "tt", "lidar_pred.npy")), lt)
    assert json.load(open(os.path.join(out2, "tt", "predict.json")))["head"] is None
    # ... and the leaderboard files: one class per raw point, both models
    exp = load_script("export_submission")
    for name in ("fused", "lidar"):
        monkeypatch.setattr(sys, "argv", ["export_submission.py", "--cache", cache, "--pred", os.path.join(out, "{zone}", f"{name}_pred.npy"),
                                          "--out", str(tmp_path / "sub" / name), "--zones", "tt", "--name", name])
        exp.main()
    inverse, keep = index["tt"]
    sub = S.read_submission(str(tmp_path / "sub" / "fused" / "tt.npz"))
    assert sub.shape == (12000,) and np.array_equal(sub, ft[inverse]) and np.array_equal(sub[keep], ft)
    assert np.array_equal(S.read_submission(str(tmp_path / "sub" / "lidar" / "tt.npz")), lt[inverse])
    man = json.load(open(str(tmp_path / "sub" / "fused" / "manifest.json")))
    assert man["name"] == "fused" and man["zones"]["tt"]["points"] == 12000 and "voxel_metrics" not in man["zones"]["tt"]
    with pytest.raises(AssertionError, match="--head needs"):
        monkeypatch.setattr(sys, "argv", ["predict_zones.py", "--cache", cache, "--litept", "x", "--backbone-ckpt", "y", "--out", out2,
                                          "--head", head_path, "--device", "cpu"])
        mod.main()
