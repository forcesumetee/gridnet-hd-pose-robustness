"""Leaderboard submission files for GridNet-HD (pure numpy: runs on the Mac and on a CPU runtime).

The leaderboard (https://huggingface.co/spaces/heig-vd-geo/GridNet-HD-Leaderboard) takes one file per test zone,
`<zone>.npz` holding the array `data` (uint8): the predicted class of EVERY point of the zone's LAS file, in file
order - what the `create_npz` helper of the leaderboard page writes with
`np.savez_compressed(npz_path, data=las["classification"].astype(np.uint8))`. We submit the 11 remapped classes
(0 pylon ... 10 building), so the "Remap classes" box of the form stays unchecked.

Our models predict one class per occupied 5 cm voxel (cache order). `points.npz` keeps `inverse` (raw point -> index
of its voxel) and `keep` (voxel -> index of its representative raw point), so a raw point takes the class of its
voxel: `pred_raw = pred_vox[inverse]`.
"""
import json
import os
from typing import Dict, Optional, Sequence

import numpy as np

TEST_ZONES = ["t1z4", "t1z5a", "t1z7", "t3z1", "t3z2", "t3z5", "t5a2", "t6z1", "t6z5"]
NUM_CLASSES = 11
CLASS_NAMES = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb", "rock_soil", "road",
               "water", "building"]


def to_raw_order(pred_vox: np.ndarray, inverse: np.ndarray, keep: Optional[np.ndarray] = None,
                 num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Per-voxel classes (cache order) -> per-raw-point classes (LAS file order), uint8. Checks that the prediction
    and the cache belong together: one prediction per voxel, every voxel referenced, classes inside the range and,
    with `keep`, that the representative point of each voxel maps back to that voxel."""
    pred_vox = np.asarray(pred_vox)
    if pred_vox.ndim != 1:
        raise ValueError(f"prediction must be one class per voxel, got shape {pred_vox.shape}")
    n_vox = len(pred_vox)
    lo, hi = int(inverse.min()), int(inverse.max())
    if lo != 0 or hi != n_vox - 1:
        raise ValueError(f"prediction has {n_vox:,} entries but the cache indexes voxels {lo:,}..{hi:,}: "
                         "prediction and points.npz are not from the same cache")
    if int(pred_vox.max()) >= num_classes:
        raise ValueError(f"predicted class {int(pred_vox.max())} outside 0..{num_classes - 1}")
    if keep is not None:
        if len(keep) != n_vox:
            raise ValueError(f"points.npz: keep has {len(keep):,} entries for {n_vox:,} voxels")
        step = max(1, n_vox // 1_000_000)                     # a sample is enough to catch a mismatched pair of arrays
        probe = np.arange(0, n_vox, step)
        if not np.array_equal(inverse[keep[probe]], probe):
            raise ValueError("points.npz: inverse[keep] is not the identity - the cache is inconsistent")
    return pred_vox.astype(np.uint8)[inverse]


def write_submission(path: str, pred_raw: np.ndarray) -> str:
    """<zone>.npz in the leaderboard format (array `data`, uint8, compressed); written to a temporary name first."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, data=np.ascontiguousarray(pred_raw, dtype=np.uint8))
    os.replace(tmp, path)
    return path


def read_submission(path: str) -> np.ndarray:
    with np.load(path) as f:
        if f.files != ["data"]:
            raise ValueError(f"{path}: expected exactly the array 'data', found {f.files}")
        return f["data"]


def class_counts(pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Points per class id; longer than num_classes when the array holds an id outside the range."""
    return np.bincount(np.asarray(pred).ravel(), minlength=num_classes)


def check_submission(path: str, n_points: int, num_classes: int = NUM_CLASSES) -> Dict:
    """Re-read a written file the way the leaderboard does and verify it: one uint8 class per LAS point, all inside
    0..num_classes-1. Returns {"points", "counts", "bytes"}; raises ValueError otherwise."""
    data = read_submission(path)
    if data.dtype != np.uint8 or data.ndim != 1:
        raise ValueError(f"{path}: data must be a 1-D uint8 array, got {data.dtype} {data.shape}")
    if len(data) != n_points:
        raise ValueError(f"{path}: {len(data):,} predictions for {n_points:,} LAS points")
    counts = class_counts(data, num_classes)
    if len(counts) > num_classes:
        raise ValueError(f"{path}: class id {len(counts) - 1} outside 0..{num_classes - 1}")
    return {"points": int(len(data)), "counts": [int(c) for c in counts], "bytes": os.path.getsize(path)}


def confusion(gt: np.ndarray, pred: np.ndarray, num_classes: int = NUM_CLASSES, ignore: int = 255,
              chunk: int = 20_000_000) -> np.ndarray:
    """(num_classes, num_classes) counts, rows = ground truth; points with the ignore label are left out."""
    if len(gt) != len(pred):
        raise ValueError(f"{len(gt):,} labels for {len(pred):,} predictions")
    cm = np.zeros(num_classes * num_classes, np.int64)
    for s in range(0, len(gt), chunk):
        g, p = gt[s:s + chunk], pred[s:s + chunk]
        k = g != ignore
        cm += np.bincount(g[k].astype(np.int64) * num_classes + p[k].astype(np.int64), minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def metrics(cm: np.ndarray, names: Sequence[str] = CLASS_NAMES) -> Dict:
    """mIoU over the classes that occur in the ground truth or in the prediction (the pooled convention of our
    evaluation), mIoU over the classes present in the ground truth, OA and per-class IoU, in percent."""
    inter = np.diag(cm).astype(np.float64); union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    present = cm.sum(1) > 0
    return {"mIoU": float(100 * np.nanmean(iou)) if (union > 0).any() else None,
            "mIoU_present": float(100 * np.mean(iou[present])) if present.any() else None,
            "OA": float(100 * inter.sum() / max(cm.sum(), 1)),
            "per_class": {n: (None if np.isnan(v) else float(100 * v)) for n, v in zip(names, iou)},
            "points": int(cm.sum())}


def export_zone(cache_dir: str, zone: str, pred_path: str, out_path: str, las_path: Optional[str] = None) -> Dict:
    """One zone: voxel prediction + points.npz -> <out_path> (leaderboard npz), verified by reading it back.
    With `las_path` the number of points is checked against the LAS header and, when the LAS has labels (a
    validation zone), the prediction is scored on the raw points. Returns the record for the manifest."""
    pred_vox = np.load(pred_path)
    with np.load(os.path.join(cache_dir, zone, "points.npz")) as p:
        if "inverse" not in p.files:
            raise ValueError(f"{cache_dir}/{zone}/points.npz has no 'inverse' array: rebuild the cache with scripts/build_cache.py")
        inverse = p["inverse"]
        keep = p["keep"] if "keep" in p.files else None
        labels_vox = p["labels"] if "labels" in p.files else None
    pred_raw = to_raw_order(pred_vox, inverse, keep)
    rec = {"zone": zone, "voxels": int(len(pred_vox)), "prediction": os.path.abspath(pred_path)}
    if labels_vox is not None and (labels_vox != 255).any():
        rec["voxel_metrics"] = metrics(confusion(labels_vox, pred_vox))
    if las_path is not None:
        from .io import las_point_count, read_las_labels
        n_las = las_point_count(las_path)
        if n_las != len(pred_raw):
            raise ValueError(f"{zone}: the LAS has {n_las:,} points, the cache was built from {len(pred_raw):,}")
        rec["las_points_checked"] = True
        gt = read_las_labels(las_path)
        if gt is not None:
            rec["raw_metrics"] = metrics(confusion(gt, pred_raw))
    write_submission(out_path, pred_raw)
    rec.update(check_submission(out_path, len(pred_raw)))
    rec["file"] = os.path.basename(out_path)
    return rec


def write_manifest(out_dir: str, records: Sequence[Dict], extra: Optional[Dict] = None) -> str:
    """manifest.json next to the npz files: what was exported from what, per zone; merged with an existing one."""
    path = os.path.join(out_dir, "manifest.json")
    old = {}
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
    zones = dict(old.get("zones", {}))
    zones.update({r["zone"]: r for r in records})
    doc = {**{k: v for k, v in old.items() if k != "zones"}, **(extra or {}),
           "format": "npz, array 'data' (uint8), one class per LAS point in file order, classes 0..10 (remapped)",
           "class_names": CLASS_NAMES, "zones": {z: zones[z] for z in sorted(zones)}}
    with open(path + ".tmp", "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(path + ".tmp", path)
    return path
