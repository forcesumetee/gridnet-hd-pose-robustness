"""Train / evaluate the 2D branch (UPerNet) on GridNet-HD image masks, and export the per-image 2D cache.

  # train (resumes from <ckpt-dir>/last.pth)
  python scripts/train_upernet.py --data /content/data --ckpt-dir $DRIVE/ckpt/upernet_v1 \
      --train-zones t5b5,t5b1 --val-zones t1z8 --iters 8000 --max-hours 3
  # export the cache (prob @1/4 + feat @1/8 per image) for some zones
  python scripts/train_upernet.py --data /content/data --ckpt-dir $DRIVE/ckpt/upernet_v1 \
      --export-cache $DRIVE/cache2d --export-zones t1z5b,t1z8,t3z4,t4z1,t5b1,t5b5

<data>/<zone>/images/*.JPG and <data>/<zone>/masks/*.png as on the Hugging Face dataset.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gridnet.train.data2d import CropDataset, class_weights_2d, list_pairs  # noqa: E402
from gridnet.train.engine2d import CLASS_NAMES, Trainer2D  # noqa: E402
from gridnet.train.upernet import UPerNet  # noqa: E402

TRAIN_ZONES = ["t1z6a", "t1z6b", "t2z5", "t3z3", "t3z6", "t3z7", "t5a1", "t5a3", "t5a4", "t5a5", "t5b2",
               "t5b3", "t5b4", "t5b6", "t5c1", "t5c2", "t5c3", "t6z2", "t6z3", "t6z4", "t6z6"]
VAL_ZONES = ["t1z5b", "t1z8", "t3z4", "t4z1", "t5b1", "t5b5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--train-zones", default=None, help="comma list; default = official train zones present in --data")
    ap.add_argument("--val-zones", default=None, help="comma list; default = official val zones present in --data")
    ap.add_argument("--images-per-zone", type=int, default=0, help="random subset of images per train zone (0 = all)")
    ap.add_argument("--val-images", type=int, default=40, help="images per val zone used for the fixed val crops")
    ap.add_argument("--val-crops-per-image", type=int, default=2)
    ap.add_argument("--encoder", default="convnext_tiny")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--scales", default="0.5,0.75,1.0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3, help="decoder LR; encoder uses --encoder-lr-mult x this")
    ap.add_argument("--encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lovasz", type=float, default=1.0)
    ap.add_argument("--class-weight-power", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--max-hours", type=float, default=1e9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--export-cache", default=None, metavar="DIR", help="write <DIR>/<zone>/<image>.npz with best.pth")
    ap.add_argument("--export-zones", default=None, help="comma list for --export-cache (default = val zones)")
    ap.add_argument("--export-stride", type=int, default=768)
    ap.add_argument("--export-limit", type=int, default=0, help="debug: only the first N images per zone")
    ap.add_argument("--export-fp16", action="store_true", help="fp16 maps (~50 MB/image) instead of the compact uint8/int8 ones (~24 MB)")
    a = ap.parse_args()

    present = set(os.listdir(a.data))
    train_names = a.train_zones.split(",") if a.train_zones else [z for z in TRAIN_ZONES if z in present]
    val_names = a.val_zones.split(",") if a.val_zones else [z for z in VAL_ZONES if z in present]
    exporting = a.export_cache is not None
    assert val_names or exporting, f"no val zones in {a.data}: {sorted(present)}"
    assert train_names or a.eval_only or exporting, f"no train zones in {a.data}: {sorted(present)}"

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = UPerNet(a.encoder, num_classes=11, pretrained=not a.no_pretrained and not (a.eval_only or exporting))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train2d] UPerNet/{a.encoder} {n_params:.1f} M params", flush=True)
    cfg = {k: v for k, v in vars(a).items() if k not in ("eval_only", "max_hours", "export_cache", "export_zones",
                                                          "export_stride", "export_limit", "export_fp16")}

    if exporting:
        tr = Trainer2D(model, None, None, cfg, a.ckpt_dir)
        name = "best.pth" if os.path.exists(os.path.join(a.ckpt_dir, "best.pth")) else "last.pth"
        assert tr.resume(name), "no checkpoint to export from"
        zones = a.export_zones.split(",") if a.export_zones else val_names
        n = tr.export_cache(a.data, zones, a.export_cache, crop=a.crop, stride=a.export_stride, batch=a.batch,
                            limit=a.export_limit or None, compact=not a.export_fp16)
        print(f"[export] wrote {n} files to {a.export_cache}", flush=True)
        return

    scales = [float(s) for s in a.scales.split(",")]
    t0 = time.time()
    val_pairs = list_pairs(a.data, val_names, per_zone=a.val_images, seed=a.seed)
    val_ds = CropDataset(a.data, val_pairs, crop=a.crop, train=False, rare_prob=0.5, seed=a.seed,
                         crops_per_image=a.val_crops_per_image)
    if a.eval_only:
        tr = Trainer2D(model, None, val_ds, cfg, a.ckpt_dir)
        name = "best.pth" if os.path.exists(os.path.join(a.ckpt_dir, "best.pth")) else "last.pth"
        assert tr.resume(name), "no checkpoint to evaluate"
        miou, ious, oa = tr.evaluate()
        with open(os.path.join(a.ckpt_dir, "eval2d.json"), "w") as fh:
            json.dump({"ckpt": name, "it": tr.it, "mIoU": miou, "OA": oa, "val_crops": len(val_ds),
                       "per_class": {n: (None if np.isnan(v) else float(100 * v)) for n, v in zip(CLASS_NAMES, ious)},
                       "confusion": tr.confusion.tolist()}, fh, indent=1)
        return

    train_pairs = list_pairs(a.data, train_names, per_zone=a.images_per_zone or None, seed=a.seed)
    print(f"[train2d] {len(train_pairs)} train images from {train_names}; {len(val_ds)} fixed val crops from {val_names}", flush=True)
    cw = class_weights_2d(a.data, train_pairs, power=a.class_weight_power, seed=a.seed)
    print("[train2d] class weights:", np.round(cw, 2).tolist(), f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    train_ds = CropDataset(a.data, train_pairs, crop=a.crop, train=True, scales=scales, rare_prob=0.5, seed=a.seed)
    tr = Trainer2D(model, train_ds, val_ds, cfg, a.ckpt_dir, class_w=cw)
    tr.resume("last.pth")
    status = tr.fit(max_hours=a.max_hours)
    print(f"[train2d] {status} at it {tr.it}, best val mIoU {tr.best:.2f}, wall {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
