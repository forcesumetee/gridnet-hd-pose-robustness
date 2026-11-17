"""Voxel predictions -> leaderboard files (one <zone>.npz per zone, array `data`, one class per LAS point).

  # test zones, fused model (predictions written by scripts/predict_zones.py)
  python scripts/export_submission.py --cache $DRIVE/cache --pred "$OUT/pred/{zone}/fused_pred.npy" --out $OUT/fused
  # the same for the LiDAR-only reference
  python scripts/export_submission.py --cache $DRIVE/cache --pred "$OUT/pred/{zone}/lidar_pred.npy" --out $OUT/lidar
  # a validation zone, scored on the raw points of its LAS file (needs laspy and <las-root>/<zone>/lidar/*.las)
  python scripts/export_submission.py --cache $DRIVE/cache --pred "$DRIVE/fusion/late_a/eval/{zone}/fused_pred.npy" \
      --out /content/check --zones t1z8 --las-root /content/data

Pure numpy (no torch). Every file is read back and checked (uint8, one value per LAS point, classes 0..10);
<out>/manifest.json records points, class counts and, for labelled zones, the scores per voxel and per raw point.
Zones whose file exists are skipped unless --force. What to upload: the nine files of ONE --out folder, in the
"Submit Eval" form of https://huggingface.co/spaces/heig-vd-geo/GridNet-HD-Leaderboard, "Remap classes" unchecked.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gridnet.submission import CLASS_NAMES, TEST_ZONES, export_zone, write_manifest  # noqa: E402


def find_las(las_root, zone):
    d = os.path.join(las_root, zone, "lidar")
    files = sorted(f for f in os.listdir(d) if f.lower().endswith((".las", ".laz"))) if os.path.isdir(d) else []
    return os.path.join(d, files[0]) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="LiDAR caches <cache>/<zone>/points.npz (with `inverse`)")
    ap.add_argument("--pred", required=True, help="voxel prediction per zone, with {zone} in the path")
    ap.add_argument("--out", required=True, help="folder for <zone>.npz + manifest.json")
    ap.add_argument("--zones", default=",".join(TEST_ZONES))
    ap.add_argument("--las-root", default=None, help="<las-root>/<zone>/lidar/*.las: check the point count; score labelled zones on raw points")
    ap.add_argument("--name", default=None, help="what these predictions are (stored in the manifest)")
    ap.add_argument("--force", action="store_true", help="rewrite files that exist")
    a = ap.parse_args()
    assert "{zone}" in a.pred, "--pred needs the placeholder {zone}"
    zones = [z for z in a.zones.split(",") if z]
    os.makedirs(a.out, exist_ok=True)
    records, missing = [], []
    for z in zones:
        pred, out = a.pred.format(zone=z), os.path.join(a.out, f"{z}.npz")
        if not os.path.exists(pred):
            missing.append(z); print(f"[export] {z}: no prediction at {pred}", flush=True); continue
        if os.path.exists(out) and not a.force:
            print(f"[export] {z}: {out} exists (skipped; --force rewrites it)", flush=True); continue
        t0 = time.time()
        las = find_las(a.las_root, z) if a.las_root else None
        if a.las_root and las is None:
            print(f"[export] {z}: no LAS under {a.las_root}/{z}/lidar - point count and raw-point score not checked", flush=True)
        rec = export_zone(a.cache, z, pred, out, las_path=las)
        records.append(rec)
        share = "  ".join(f"{n} {100 * c / rec['points']:.2f}" for n, c in zip(CLASS_NAMES, rec["counts"]) if c)
        print(f"[export] {z}: {rec['points']:,} points from {rec['voxels']:,} voxels -> {out} ({rec['bytes'] / 1e6:.1f} MB, "
              f"{time.time() - t0:.0f} s)\n         predicted classes [%]: {share}", flush=True)
        for key, label in (("voxel_metrics", "per voxel "), ("raw_metrics", "per raw point")):
            if key in rec:
                m = rec[key]
                print(f"         {label}: mIoU {m['mIoU_present']:.2f} (classes present)  OA {m['OA']:.2f}  over {m['points']:,} labelled", flush=True)
    path = write_manifest(a.out, records, extra={"name": a.name} if a.name else None)
    print(f"[export] {len(records)} file(s) written, manifest {path}" + (f"; NO prediction for {missing}" if missing else ""), flush=True)
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
