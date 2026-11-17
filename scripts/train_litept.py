"""Train LitePT-S (LiDAR-only) on cached GridNet-HD zones. Resume-safe; made for Colab sessions.

  python scripts/train_litept.py --cache /content/cache --ckpt-dir /content/drive/MyDrive/GridNet/ckpt/litept_s_lidar \
      --litept /content/LitePT --iters 20000 --max-hours 10

Re-running the same command resumes from <ckpt-dir>/last.pth. Zone split comes from the dataset's
split.json (train 21 / val 6) unless --train-zones / --val-zones are given.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gridnet.train.data import FEATURE_SETS, class_weights, load_zones  # noqa: E402
from gridnet.train.engine import Trainer, build_model  # noqa: E402

TRAIN_ZONES = ["t1z6a", "t1z6b", "t2z5", "t3z3", "t3z6", "t3z7", "t5a1", "t5a3", "t5a4", "t5a5", "t5b2",
               "t5b3", "t5b4", "t5b6", "t5c1", "t5c2", "t5c3", "t6z2", "t6z3", "t6z4", "t6z6"]
VAL_ZONES = ["t1z5b", "t1z8", "t3z4", "t4z1", "t5b1", "t5b5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="dir with <zone>/points.npz")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--litept", default="/content/LitePT")
    ap.add_argument("--train-zones", default=None, help="comma list; default = official train split (those present)")
    ap.add_argument("--val-zones", default=None, help="comma list; default = official val split (those present)")
    ap.add_argument("--features", default="geo", choices=list(FEATURE_SETS))
    ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--max-points", type=int, default=600_000,
                    help="cap per training crop (nearest-N around the centre). ~18 KB VRAM per point incl. "
                         "backward: 2x600k fits 80 GB; use 300k on a 24 GB L4")
    ap.add_argument("--eval-max-points", type=int, default=900_000,
                    help="cap per eval window (fp32, no grad); 500k on L4")
    ap.add_argument("--rare-prob", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="base LR (embedding/pooling/head). LitePT nuScenes recipe is 2e-3 at batch 12; "
                         "2e-3 at batch 2x600k diverged (NaN at it ~2350), 1e-3 is the default")
    ap.add_argument("--block-lr-mult", type=float, default=0.1,
                    help="LR multiplier for parameters with 'block' in their name (LitePT/PTv3 param_dicts)")
    ap.add_argument("--wd", type=float, default=0.005)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lovasz", type=float, default=1.0)
    ap.add_argument("--class-weight-power", type=float, default=0.5)
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--max-hours", type=float, default=1e9, help="stop (with checkpoint) after this many hours")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="evaluate best.pth (or last.pth) on the val zones")
    ap.add_argument("--save-pred", default=None, metavar="DIR",
                    help="with --eval-only: write per-point predictions <DIR>/<zone>_pred.npy (uint8, cache order) "
                         "for qualitative figures (scripts/render_predictions.py)")
    a = ap.parse_args()

    present = set(os.listdir(a.cache))
    train_names = a.train_zones.split(",") if a.train_zones else [z for z in TRAIN_ZONES if z in present]
    val_names = a.val_zones.split(",") if a.val_zones else [z for z in VAL_ZONES if z in present]
    assert val_names and (train_names or a.eval_only), f"zones missing in {a.cache}: {sorted(present)}"
    print(f"[train] train zones ({len(train_names)}): {train_names}\n[train] val zones ({len(val_names)}): {val_names}", flush=True)

    t0 = time.time()
    # eval-only needs no training zones (saves ~4 min and ~20 GB RAM on the full split)
    train_zones = [] if a.eval_only else load_zones(a.cache, train_names, a.features)
    val_zones = load_zones(a.cache, val_names, a.features)
    print(f"[train] loaded {sum(z.n for z in train_zones):,} train / {sum(z.n for z in val_zones):,} val points "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)

    in_ch = 3 + len(FEATURE_SETS[a.features])
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = build_model(a.litept, in_ch, drop_path=a.drop_path)
    print(f"[train] LitePT-S in_ch={in_ch}, params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)

    cfg = {k: v for k, v in vars(a).items() if k not in ("eval_only", "max_hours", "save_pred")}
    cfg["train_zones"], cfg["val_zones"] = train_names, val_names
    cw = class_weights(train_zones, a.class_weight_power) if train_zones else None
    if cw is not None:
        print("[train] class weights:", np.round(cw, 2).tolist(), flush=True)
    tr = Trainer(model, train_zones, val_zones, cfg, a.ckpt_dir, class_w=cw)

    if a.eval_only:
        name = "best.pth" if os.path.exists(os.path.join(a.ckpt_dir, "best.pth")) else "last.pth"
        assert tr.resume(name), "no checkpoint to evaluate"
        miou, ious, oa = tr.evaluate(save_dir=a.save_pred)
        names = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg",
                 "herb", "rock_soil", "road", "water", "building"]
        pc = lambda v: {n: (None if np.isnan(x) else float(100 * x)) for n, x in zip(names, v)}  # noqa: E731
        with open(os.path.join(a.ckpt_dir, "eval.json"), "w") as fh:
            json.dump({"ckpt": name, "it": tr.it, "mIoU": miou, "OA": oa, "per_class": pc(ious),
                       "per_zone": {z: {"mIoU": r["mIoU"], "OA": r["OA"], "per_class": pc(r["iou"])}
                                    for z, r in tr.per_zone.items()},
                       "confusion": tr.confusion.tolist(),   # rows = ground truth, cols = prediction
                       "class_names": names}, fh, indent=1)
        return

    tr.resume("last.pth")
    status = tr.fit(max_hours=a.max_hours)
    print(f"[train] {status} at it {tr.it}, best val mIoU {tr.best:.2f}, wall {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
