"""Train / evaluate a fusion head on frozen backbones (LitePT-S LiDAR-only checkpoint + UPerNet 2D cache).

  python scripts/train_fusion.py --cache /content/cache --cache2d /content/cache2d --data /content/data \
      --assoc-dir $DRIVE/fusion/assoc --litept /content/LitePT --backbone-ckpt $DRIVE/ckpt/litept_s_lidar_v2/best.pth \
      --work-dir $DRIVE/fusion/concat_v1 --head concat \
      --train-zones t1z6a,t2z5,t3z3,t5a1,t5b2,t5c1,t6z2,t6z4 --val-zones t1z5b,t1z8,t3z4,t4z1,t5b1,t5b5

Heads: late (LiDAR probs ⊕ image probs), concat (LiDAR feature ⊕ image feature), window (K×K window attention).
Pose-noise augmentation: --pose-noise-t 0.25 --pose-noise-r 0.25 (max translation m / rotation deg per crop).
Resumes from <work-dir>/last.pth; --eval-only evaluates best.pth and writes <work-dir>/eval/eval.json.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gridnet.train.engine_fusion import (FrozenBackbone, FusionTrainer, class_weights_from_counts,  # noqa: E402
                                         label_counts, load_zone_context)
from gridnet.train.fusion import build_head  # noqa: E402

VAL_ZONES = ["t1z5b", "t1z8", "t3z4", "t4z1", "t5b1", "t5b5"]
TRAIN_DEFAULT = ["t1z6a", "t2z5", "t3z3", "t5a1", "t5b2", "t5c1", "t6z2", "t6z4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="LiDAR caches <cache>/<zone>/points.npz")
    ap.add_argument("--cache2d", required=True, help="2D cache <cache2d>/<zone>/<image>.npz (one UPerNet run)")
    ap.add_argument("--data", required=True, help="<data>/<zone>/pose/* camera files")
    ap.add_argument("--assoc-dir", required=True, help="<assoc-dir>/<zone>/assoc.npz (computed + saved when missing)")
    ap.add_argument("--assoc-source", choices=["auto", "archive", "compute"], default="auto",
                    help="auto = rebuild the association on a GPU (faster than reading the archive from Drive), read the archive on CPU")
    ap.add_argument("--litept", required=True); ap.add_argument("--backbone-ckpt", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--head", default="concat", choices=["late", "concat", "window"])
    ap.add_argument("--k", type=int, default=3, help="window size for --head window")
    ap.add_argument("--lidar-input", choices=["feat", "prob"], default="feat",
                    help="concat / window: what the head sees of the LiDAR branch (72-d feature or class probabilities)")
    ap.add_argument("--image-input", choices=["feat", "prob"], default="feat",
                    help="concat: image feature + probabilities, or probabilities only")
    ap.add_argument("--lidar-drop", type=float, default=0.0, help="LiDAR modality dropout during head training (fraction of points)")
    ap.add_argument("--uniform-views", action="store_true", help="late head: uniform view weights instead of learned")
    ap.add_argument("--train-zones", default=",".join(TRAIN_DEFAULT))
    ap.add_argument("--val-zones", default=",".join(VAL_ZONES))
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--its-per-zone", type=int, default=300)
    ap.add_argument("--head-points", type=int, default=120_000, help="labelled points per crop used for the head loss")
    ap.add_argument("--head-chunk", type=int, default=50_000)
    ap.add_argument("--radius", type=float, default=25.0); ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--max-points", type=int, default=600_000); ap.add_argument("--eval-max-points", type=int, default=900_000)
    ap.add_argument("--rare-prob", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lovasz", type=float, default=1.0); ap.add_argument("--class-weight-power", type=float, default=0.5)
    ap.add_argument("--pose-noise-t", type=float, default=0.0, help="max camera translation noise (m) for augmentation")
    ap.add_argument("--pose-noise-r", type=float, default=0.0, help="max camera rotation noise (deg) for augmentation")
    ap.add_argument("--aug-prob", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--assoc-chunk", type=int, default=20_000_000)
    ap.add_argument("--log-every", type=int, default=20); ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--no-eval-each-pass", action="store_true")
    ap.add_argument("--max-hours", type=float, default=1e9); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-name", default="eval", help="sub-folder of --work-dir for the evaluation outputs")
    a = ap.parse_args()
    if a.eval_only:                 # the head design is whatever the checkpoint was trained with (also bank-trained heads)
        for name in ("best.pth", "last.pth"):
            path = os.path.join(a.work_dir, name)
            if os.path.exists(path):
                saved = torch.load(path, map_location="cpu", weights_only=False)["cfg"]
                a.head, a.k = saved.get("head", a.head), saved.get("k", a.k)
                a.lidar_input = saved.get("lidar_input", "feat"); a.image_input = saved.get("image_input", "feat")
                a.uniform_views = bool(saved.get("uniform_views", False))
                break

    train_names = [z for z in a.train_zones.split(",") if z]
    val_names = [z for z in a.val_zones.split(",") if z]
    cfg = {k: v for k, v in vars(a).items() if k not in ("eval_only", "max_hours", "device", "no_eval_each_pass", "eval_name")}
    cfg.update(train_zones=train_names, val_zones=val_names, eval_every_pass=not a.no_eval_each_pass)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    head = build_head(a.head, k=a.k, lidar_input=a.lidar_input, image_input=a.image_input,
                      **({"uniform_views": True} if (a.head == "late" and a.uniform_views) else {}))
    print(f"[fusion] head {a.head}" + (f" K={a.k}" if a.head == "window" else "") +
          f": {sum(p.numel() for p in head.parameters()) / 1e3:.1f} k parameters", flush=True)
    backbone = FrozenBackbone(a.litept, a.backbone_ckpt, in_channels=7, device=a.device)

    def loader(zone, need_depths):
        return load_zone_context(a.cache, a.cache2d, a.data, a.assoc_dir, zone, k=a.views, tau=a.tau, chunk=a.assoc_chunk,
                                 need_depths=need_depths, with_feat=head.uses_feat, device=a.device,
                                 assoc_source=a.assoc_source)

    class_w = None
    if not a.eval_only:
        class_w = class_weights_from_counts(label_counts(a.cache, train_names), a.class_weight_power)
        print("[fusion] class weights:", [round(float(w), 2) for w in class_w], flush=True)
    tr = FusionTrainer(backbone, head, cfg, a.work_dir, class_w=class_w, device=a.device)
    save_dir = os.path.join(a.work_dir, a.eval_name)

    if a.eval_only:
        name = "best.pth" if os.path.exists(os.path.join(a.work_dir, "best.pth")) else "last.pth"
        assert tr.resume(name, weights_only=True), "no checkpoint to evaluate"
        tr.evaluate(loader, val_names, save_dir=save_dir, tag=name[:-4])
        return

    last = os.path.join(a.work_dir, "last.pth")
    if os.path.exists(last) and torch.load(last, map_location="cpu", weights_only=False)["cfg"].get("trainer") == "bank":
        raise SystemExit(f"[fusion] {a.work_dir} holds a head trained by train_heads_bank.py: evaluate it with --eval-only "
                         "(full-zone evaluation) or choose another --work-dir to train a new head")
    tr.resume("last.pth")
    t0 = time.time()
    status = tr.fit(loader, val_names, max_hours=a.max_hours,
                    eval_fn=lambda t: t.evaluate(loader, val_names, save_dir=save_dir, tag="pass")["mIoU"])
    print(f"[fusion] {status} at it {tr.it}, best val mIoU {tr.best:.2f}, wall {(time.time() - t0) / 3600:.2f} h", flush=True)
    if status == "done":
        summ, redone = tr.finish(loader, val_names, save_dir)
        if not redone:
            print(f"[fusion] eval/ already holds the evaluation of best.pth (it {summ['it']}, mIoU {summ['mIoU']:.2f}, "
                  f"LiDAR-only {summ['lidar_only']['mIoU']:.2f}) - no re-evaluation", flush=True)


if __name__ == "__main__":
    main()
