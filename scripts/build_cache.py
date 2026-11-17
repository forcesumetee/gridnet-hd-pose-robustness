"""Build the per-zone cache used by every later experiment.

  python scripts/build_cache.py --zone data/t5b5 --out data/cache --voxel 0.05 --k 8

Outputs  <out>/<zone>/points.npz  xyz (float32, relative to `origin`), labels (uint8, 11-class),
                                  linearity/planarity/sphericity/verticality/hag (float32),
                                  normals (float16), inverse (int32: raw point -> voxel index),
                                  labeled (bool: False for a test zone, whose LAS has no ground_truth
                                  field; its labels are all 255)
         <out>/<zone>/assoc.npz   K best views per voxel point (see gridnet.preprocess.association)
         <out>/<zone>/report.txt  class counts + view coverage
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gridnet.classes import CLASSES  # noqa: E402
from gridnet.camera import load_zone_cameras  # noqa: E402
from gridnet.io import ensure_dir, load_zone_points  # noqa: E402
from gridnet.preprocess.association import build_association, coverage_report  # noqa: E402
from gridnet.preprocess.geometry import height_above_ground, majority_labels, pca_features, voxel_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True)
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--voxel", type=float, default=0.05, help="grid size in metres")
    ap.add_argument("--k", type=int, default=8, help="views kept per point")
    ap.add_argument("--knn", type=int, default=16, help="neighbours for PCA features")
    ap.add_argument("--feat-grid", type=float, default=0.10,
                    help="compute PCA features on this coarser grid and map back (0 = on every point)")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--no-assoc", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    zone = os.path.basename(os.path.normpath(args.zone))
    out_dir = ensure_dir(os.path.join(args.out, zone))

    pts = load_zone_points(args.zone)
    xyz_raw, labels_raw = pts["xyz"], pts["labels"]
    labeled = bool(pts.get("labeled", True))
    print(f"[cache] {zone}: {len(labels_raw):,} raw points" + ("" if labeled else "  (no ground_truth field: unlabeled test zone)"))

    keep, inverse = voxel_grid(xyz_raw, args.voxel)
    xyz = xyz_raw[keep]
    labels = majority_labels(labels_raw, inverse, len(keep))
    print(f"[cache] voxel {args.voxel} m -> {len(keep):,} points ({len(keep) / len(labels_raw) * 100:.1f} %)")

    if args.feat_grid > args.voxel:
        keep2, inv2 = voxel_grid(xyz, args.feat_grid)
        print(f"[cache] PCA features on {len(keep2):,} points at {args.feat_grid} m, mapped back to {len(xyz):,}")
        geo = {k: v[inv2] for k, v in pca_features(xyz[keep2], k=args.knn).items()}
    else:
        geo = pca_features(xyz, k=args.knn)
    hag = height_above_ground(xyz)
    origin = xyz.min(axis=0)
    np.savez(os.path.join(out_dir, "points.npz"), xyz=(xyz - origin).astype(np.float32), origin=origin,
             labels=labels, inverse=inverse.astype(np.int32), keep=keep.astype(np.int32),
             linearity=geo["linearity"].astype(np.float16), planarity=geo["planarity"].astype(np.float16),
             sphericity=geo["sphericity"].astype(np.float16), verticality=geo["verticality"].astype(np.float16),
             normals=geo["normals"].astype(np.float16), hag=hag.astype(np.float16), voxel=args.voxel,
             labeled=np.array(labeled))

    lines = [f"zone {zone}  raw {len(labels_raw):,}  voxel {args.voxel} m -> {len(keep):,}"
             + ("" if labeled else "  (unlabeled test zone)"), ""]
    tot = len(labels)
    for c in sorted(set(np.unique(labels).tolist())):
        n = int((labels == c).sum())
        lines.append(f"{c:>4} {CLASSES.get(c, '?'):<26}{n:>12,}{100 * n / tot:>8.3f}%")

    if not args.no_assoc:
        cal, poses = load_zone_cameras(args.zone)
        assoc = build_association(xyz, poses, cal, k=args.k, normals=geo["normals"].astype(np.float64),
                                  ds=args.ds, tau=args.tau)
        assoc.save(os.path.join(out_dir, "assoc.npz"))
        lines += ["", f"views per point (K={args.k}):", coverage_report(assoc, labels, CLASSES)]

    lines.append(f"\nelapsed {time.time() - t0:.0f} s")
    report = "\n".join(lines)
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()
