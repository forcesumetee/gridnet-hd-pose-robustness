"""Export a compact smoke-test file from a zone cache for Colab (no raw LAS needed there).

  python scripts/export_smoke.py --cache data/cache/t5b5 --out data/smoke_t5b5.npz --grid 0.10

Re-grids the 5 cm cache to `--grid` and stores xyz (float32, zone-relative), labels (uint8),
and per-point features linearity/planarity/verticality/hag (float16). ~400 MB for a zone at 10 cm.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gridnet.preprocess.geometry import majority_labels, voxel_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="data/cache/<zone>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=float, default=0.10)
    a = ap.parse_args()

    p = np.load(os.path.join(a.cache, "points.npz"))
    xyz, labels = p["xyz"].astype(np.float64), p["labels"]
    keep, inv = voxel_grid(xyz, a.grid)
    lab = majority_labels(labels, inv, len(keep))
    feats = np.stack([p["linearity"][keep], p["planarity"][keep], p["verticality"][keep], p["hag"][keep]], axis=1)
    np.savez_compressed(a.out, xyz=xyz[keep].astype(np.float32), labels=lab, feats=feats.astype(np.float16),
                        feat_names=np.array(["linearity", "planarity", "verticality", "hag"]), grid=a.grid,
                        zone=os.path.basename(os.path.normpath(a.cache)))
    print(f"{len(labels):,} -> {len(keep):,} points at {a.grid} m; wrote {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
