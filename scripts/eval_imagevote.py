"""ImageVote on the validation zones: GPU association (K nearest visible views) + averaging of the cached
2D class probabilities per point. Writes per-zone predictions / image probabilities / association tables
and an eval.json-compatible summary (so scripts/plot_eval.py can draw it).

  python scripts/eval_imagevote.py --cache /content/cache --data /content/data --cache2d $DRIVE/cache2d \
      --zones t1z5b,t1z8,t3z4,t4z1,t5b1,t5b5 --out $DRIVE/fusion/imagevote --save-assoc

<cache>/<zone>/points.npz (xyz, origin, labels, normals), <data>/<zone>/pose/* (camera files),
<cache2d>/<zone>/<image>.npz (prob @1/4). Points without any visible view get --fallback (6 = herbaceous).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gridnet.camera import load_zone_cameras  # noqa: E402
from gridnet.preprocess.association import Association, coverage_report  # noqa: E402
from gridnet.classes import CLASSES  # noqa: E402
from gridnet.train.assoc_gpu import associate  # noqa: E402
from gridnet.train.data import ConfusionMeter  # noqa: E402
from gridnet.train.imagevote import vote  # noqa: E402
from gridnet.train.maps2d import cached_image_ids  # noqa: E402

NAMES = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb", "rock_soil", "road", "water", "building"]


def per_class(ious):
    return {n: (None if np.isnan(v) else float(100 * v)) for n, v in zip(NAMES, ious)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--cache2d", required=True); ap.add_argument("--zones", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=20_000_000, help="points per GPU chunk (lower on a 24 GB GPU: 12e6)")
    ap.add_argument("--fallback", type=int, default=6, help="class for points without any view")
    ap.add_argument("--save-assoc", action="store_true", help="write <out>/<zone>/assoc.npz (~6 GB per 50 M points)")
    ap.add_argument("--reuse-assoc", action="store_true", help="load <out>/<zone>/assoc.npz if present")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    zones = [z.strip() for z in a.zones.split(",") if z.strip()]
    total = ConfusionMeter(); covered = ConfusionMeter()
    summary = {"model": "ImageVote", "k": a.k, "tau": a.tau, "per_zone": {}}
    for z in zones:
        t0 = time.time()
        zdir = os.path.join(a.out, z); os.makedirs(zdir, exist_ok=True)
        with np.load(os.path.join(a.cache, z, "points.npz")) as p:
            xyz, origin, labels = p["xyz"], p["origin"], p["labels"].astype(np.uint8)
            normals = p["normals"] if "normals" in p else None
        cal, poses = load_zone_cameras(os.path.join(a.data, z))
        have = set(cached_image_ids(a.cache2d, z))
        image_ids = sorted(i for i in poses if i in have)
        print(f"[{z}] {len(xyz):,} points, {len(image_ids)} images with 2D cache ({len(poses)} poses, {len(have)} cached)", flush=True)
        apath = os.path.join(zdir, "assoc.npz")
        if a.reuse_assoc and os.path.exists(apath):
            assoc = Association.load(apath); print(f"[{z}] association loaded from {apath}")
        else:
            assoc = associate(xyz, origin, poses, cal, image_ids, k=a.k, tau=a.tau, normals=normals,
                              chunk=a.chunk, device=a.device)
            if a.save_assoc:
                assoc.save(apath)
        t1 = time.time()
        print(f"[{z}] association {(t1 - t0) / 60:.1f} min; views/point mean {assoc.n_views.mean():.2f}, "
              f"no view {100 * (assoc.n_views == 0).mean():.1f} %", flush=True)
        print(coverage_report(assoc, labels, CLASSES), flush=True)
        prob, counts = vote(assoc, a.cache2d, z, device=a.device)
        pred = prob.argmax(1).astype(np.uint8)
        pred[counts == 0] = a.fallback
        np.save(os.path.join(zdir, "imagevote_pred.npy"), pred)
        np.save(os.path.join(zdir, "img_prob.npy"), prob.astype(np.float16))
        np.save(os.path.join(zdir, "n_views.npy"), counts.astype(np.uint8))
        zm = ConfusionMeter(); zm.add(labels, pred); total.m += zm.m
        cm = ConfusionMeter(); cm.add(labels[counts > 0], pred[counts > 0]); covered.m += cm.m
        lab = labels != 255
        summary["per_zone"][z] = {"mIoU": 100 * zm.miou(present_only=True), "OA": 100 * zm.oa(), "per_class": per_class(zm.iou()),
                                  "mIoU_covered": 100 * cm.miou(present_only=True), "coverage": float(100 * (counts[lab] > 0).mean()),
                                  "images": len(image_ids), "points": int(len(xyz)),
                                  "assoc_min": (t1 - t0) / 60, "vote_min": (time.time() - t1) / 60}
        print(f"[{z}] ImageVote mIoU {100 * zm.miou(present_only=True):.2f} (classes present in the zone)  OA {100 * zm.oa():.2f}  "
              f"(covered points only: mIoU {100 * cm.miou(present_only=True):.2f}, coverage {summary['per_zone'][z]['coverage']:.1f} %)  "
              f"vote {(time.time() - t1) / 60:.1f} min", flush=True)
        print("     " + "  ".join(f"{n} {100 * v:.1f}" for n, v in zip(NAMES, zm.iou()) if not np.isnan(v)), flush=True)
    summary.update({"mIoU": 100 * total.miou(), "OA": 100 * total.oa(), "per_class": per_class(total.iou()),
                    "mIoU_covered": 100 * covered.miou(), "confusion": total.m.tolist(), "class_names": NAMES, "it": 0})
    with open(os.path.join(a.out, "eval.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[all] ImageVote mIoU {summary['mIoU']:.2f}  OA {summary['OA']:.2f}  (covered: {summary['mIoU_covered']:.2f})", flush=True)
    print("      " + "  ".join(f"{n} {100 * v:.1f}" for n, v in zip(NAMES, total.iou()) if not np.isnan(v)), flush=True)


if __name__ == "__main__":
    main()
