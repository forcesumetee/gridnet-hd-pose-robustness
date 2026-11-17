"""Voxel-level predictions for any zones, labelled or not (the leaderboard's test zones): the frozen LiDAR-only
model and, with --head, a fusion head. Same windows, votes and seed as the full-zone evaluation of train_fusion.py.

  # fused + LiDAR-only (needs the 2D cache and the camera files of the zones)
  python scripts/predict_zones.py --cache /content/cache --cache2d /content/cache2d --data /content/data \
      --assoc-dir /content/assoc --litept /content/LitePT --backbone-ckpt $DRIVE/ckpt/litept_s_lidar_v2/best.pth \
      --head $DRIVE/fusion/late_a/best.pth --out $OUT/pred --zones t1z4,t1z5a
  # LiDAR-only (no images)
  python scripts/predict_zones.py --cache /content/cache --litept /content/LitePT --backbone-ckpt ... --out $OUT/pred

Writes <out>/<zone>/lidar_pred.npy and, with --head, <out>/<zone>/fused_pred.npy (uint8, cache voxel order) plus
<out>/<zone>/predict.json (voxels, minutes, predicted class counts; for a labelled zone also mIoU / OA per voxel, the
numbers of eval.json). A zone whose files exist is skipped (resumable). scripts/export_submission.py turns the
predictions into the leaderboard files.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gridnet.submission import TEST_ZONES  # noqa: E402
from gridnet.train.data import load_zone  # noqa: E402
from gridnet.train.engine_fusion import FrozenBackbone, FusionTrainer, load_zone_context  # noqa: E402
from gridnet.train.predict import head_from_checkpoint, predict_lidar, summarize_prediction, zone_is_labeled  # noqa: E402


def save_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def cache_flag(cache_dir, zone):
    with np.load(os.path.join(cache_dir, zone, "points.npz")) as p:
        return bool(p["labeled"]) if "labeled" in p.files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="LiDAR caches <cache>/<zone>/points.npz")
    ap.add_argument("--litept", required=True); ap.add_argument("--backbone-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zones", default=",".join(TEST_ZONES))
    ap.add_argument("--head", default=None, help="best.pth of a fusion head; without it only the LiDAR-only prediction is made")
    ap.add_argument("--cache2d", default=None, help="with --head: 2D cache <cache2d>/<zone>/<image>.npz")
    ap.add_argument("--data", default=None, help="with --head: <data>/<zone>/pose/* camera files")
    ap.add_argument("--assoc-dir", default=None, help="with --head: <assoc-dir>/<zone>/assoc.npz (computed + saved when missing)")
    ap.add_argument("--assoc-source", choices=["auto", "archive", "compute"], default="auto")
    ap.add_argument("--radius", type=float, default=25.0); ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--eval-max-points", type=int, default=900_000, help="cap per window (900k on an A100, 500k on an L4)")
    ap.add_argument("--head-chunk", type=int, default=50_000)
    ap.add_argument("--views", type=int, default=8); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--assoc-chunk", type=int, default=20_000_000)
    ap.add_argument("--force", action="store_true", help="predict again zones that have their files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    zones = [z for z in a.zones.split(",") if z]
    fused = a.head is not None
    if fused:
        assert a.cache2d and a.data and a.assoc_dir, "--head needs --cache2d, --data and --assoc-dir"
    os.makedirs(a.out, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if str(a.device).startswith("cuda") else "cpu"
    cfg = {"radius": a.radius, "grid": a.grid, "eval_max_points": a.eval_max_points, "head_chunk": a.head_chunk,
           "views": a.views, "tau": a.tau, "passes": 1, "train_zones": [], "its_per_zone": 1, "lr": 1e-3, "seed": 0}

    backbone = FrozenBackbone(a.litept, a.backbone_ckpt, in_channels=7, device=a.device)
    tr, head_info = None, None
    if fused:
        head, head_cfg, head_info = head_from_checkpoint(a.head)
        cfg.update(head=head_cfg["head"], k=head_cfg.get("k", 3))
        print(f"[predict] head {head_cfg['head']}" + (f" K={head_cfg.get('k')}" if head_cfg["head"] == "window" else "")
              + f" from {a.head} (it {head_info['it']}, val {head_info['best'] if head_info['best'] is not None else float('nan'):.2f})", flush=True)
        tr = FusionTrainer(backbone, head, cfg, a.out, device=a.device)
    names = ["lidar"] + (["fused"] if fused else [])
    t_all, done, ctx = time.time(), [], None
    for z in zones:
        zdir = os.path.join(a.out, z)
        rec_path = os.path.join(zdir, "predict.json")
        if not a.force and os.path.exists(rec_path) and all(os.path.exists(os.path.join(zdir, f"{n}_pred.npy")) for n in names):
            print(f"[predict] {z}: predictions exist (skipped)", flush=True); done.append(z); continue
        os.makedirs(zdir, exist_ok=True)
        t0 = time.time()
        if fused:
            ctx = load_zone_context(a.cache, a.cache2d, a.data, a.assoc_dir, z, k=a.views, tau=a.tau, chunk=a.assoc_chunk,
                                    need_depths=False, with_feat=tr.head.uses_feat, device=a.device, assoc_source=a.assoc_source)
            zone = ctx.zone
            pred, lpred = tr.predict_zone(ctx)
            n_img, no_view = len(ctx.assoc.image_ids), float((ctx.assoc.n_views == 0).mean())
            ctx.free()
            preds = {"lidar": lpred, "fused": pred}
        else:
            zone = load_zone(a.cache, z)
            preds = {"lidar": predict_lidar(backbone, zone, cfg, a.device)}
            n_img, no_view = None, None
        labeled = zone_is_labeled(zone.labels, cache_flag(a.cache, z))
        rec = {"zone": z, "voxels": int(zone.n), "labeled": labeled, "minutes": (time.time() - t0) / 60, "gpu": gpu,
               "eval_max_points": a.eval_max_points, "backbone": a.backbone_ckpt, "head": a.head,
               "head_it": head_info["it"] if head_info else None, "images": n_img, "no_view_fraction": no_view}
        for n in names:
            save_npy(os.path.join(zdir, f"{n}_pred.npy"), preds[n])
            rec[n] = summarize_prediction(preds[n], zone.labels, labeled)
        with open(rec_path + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(rec_path + ".tmp", rec_path)
        line = f"[predict] {z}: {zone.n:,} voxels in {rec['minutes']:.1f} min"
        if labeled:
            line += "  " + "  ".join(f"{n} mIoU {rec[n]['metrics']['mIoU_present']:.2f} OA {rec[n]['metrics']['OA']:.2f}" for n in names)
        else:
            line += "  (no labels)"
        print(line, flush=True)
        done.append(z)
        ctx = zone = preds = None                     # the association table of a large zone is ~10 GB: free it before the next one
        if str(a.device).startswith("cuda"):
            torch.cuda.empty_cache()
    print(f"[predict] {len(done)}/{len(zones)} zones in {(time.time() - t_all) / 60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
