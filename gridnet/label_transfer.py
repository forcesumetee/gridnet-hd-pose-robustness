"""Model-free label transfer under camera-pose error (numpy: runs on the Mac and on a CPU runtime).

The question behind the motivating figure of the paper, asked without any network: the points that a camera sees
receive the label of the pixel they project to in the authors' 2D mask of that image. With the delivered pose
almost every point receives its own label; with a wrong pose the mask stays where it is and the points move over it.

One evaluation = one image and one pose:
  candidates   points near the view cone of the DELIVERED pose (`candidates`: within `max_dist`, inside the cone the
               association uses and inside the frame enlarged by `margin`); the same set serves every perturbed pose
  visible      z-buffer of the pose under test over the candidates (the test of the association stage: depth map at
               1/ds of the image resolution, tolerance tau) -> the points this, possibly wrong, camera sees
  per class c  hits[c] / total[c]: visible points of class c whose pixel carries label c. A point that lands on an
               unlabelled pixel (a cable projected onto the sky) is a miss.
  all          hits over the visible labelled points that land on a labelled pixel

Error models (`noise`): "protocol" = the i.i.d. model of the robustness protocol (camera centre + N(0, sigma_t^2 I),
rotation by an axis-angle vector with per-axis angles N(0, sigma_r^2)); "fixed" = a displacement of fixed length in
a random direction, or a rotation by a fixed angle about a random axis (the first, single-image experiment).
Every draw has its own seed, a function of (seed, image id, kind, level, draw): a subset of the images, another
order of the images or a resumed run give the same numbers.
"""
import csv
import os
import zlib
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .camera import Calibration, Pose, frustum_prefilter, perturb_pose, perturb_pose_gaussian, project, visible

NUM_CLASSES = 11
CLASS_NAMES = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb", "rock_soil", "road",
               "water", "building"]
KINDS = {"none": 0, "translation_cm": 1, "rotation_deg": 2}
TRANS_CM = (2.0, 5.0, 10.0, 25.0, 50.0)
ROT_DEG = (0.05, 0.1, 0.25, 0.5, 1.0)


def draw_rng(seed: int, image_id: str, kind: str, level: float, draw: int) -> np.random.Generator:
    """Generator of one draw; depends on the image NAME and the level VALUE, never on their position in a list."""
    return np.random.default_rng(np.random.SeedSequence(
        [int(seed), zlib.crc32(image_id.encode()), KINDS[kind], int(round(level * 1e6)), int(draw)]))


def perturbed(pose: Pose, kind: str, level: float, rng: np.random.Generator, noise: str = "protocol") -> Pose:
    """The pose under test: `level` is centimetres for "translation_cm" and degrees for "rotation_deg"."""
    if kind == "none" or level == 0:
        return pose
    t, r = (level / 100.0, 0.0) if kind == "translation_cm" else (0.0, level)
    if noise == "protocol":
        return perturb_pose_gaussian(pose, rng, sigma_t=t, sigma_r_deg=r)
    if noise == "fixed":
        return perturb_pose(pose, rng, rot_deg=r, trans_m=t)
    raise ValueError(f"unknown noise model {noise!r}")


def candidates(xyz: np.ndarray, pose: Pose, cal: Calibration, max_dist: float = 150.0, margin: float = 0.3) -> np.ndarray:
    """Indices of the points that any pose near `pose` can see: closer than `max_dist`, inside the view cone of the
    association stage and inside the frame enlarged by `margin` on every side (in normalised image coordinates;
    `project` itself accepts nothing beyond 1.15 half-frames). Cheap float32 distance test first, so that a zone of
    tens of millions of points costs a fraction of a second per image."""
    S32 = pose.S.astype(np.float32)
    d2 = (xyz[:, 0] - S32[0]) ** 2 + (xyz[:, 1] - S32[1]) ** 2
    near = np.flatnonzero(d2 < np.float32(max_dist) ** 2)
    if not len(near):
        return near
    sub = xyz[near].astype(np.float64)
    keep = frustum_prefilter(sub, pose, cal, max_dist=max_dist)
    P = (sub[keep] - pose.S) @ pose.R.T
    tx, ty = cal.half_fov_tan
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = (np.abs(P[:, 0] / P[:, 2]) <= (1 + margin) * tx) & (np.abs(P[:, 1] / P[:, 2]) <= (1 + margin) * ty)
    return near[keep[inside]]


def transfer_counts(xyz: np.ndarray, labels: np.ndarray, mask: np.ndarray, pose: Pose, cal: Calibration,
                    ds: int = 4, tau: float = 0.5):
    """One image, one pose -> (hits (11,), total (11,), hits_all, n_all); see the module docstring."""
    u, v, z, ok = project(xyz, pose, cal)
    vis, _ = visible(u, v, z, ok, cal, ds=ds, tau=tau)
    ui, vi = u[vis].astype(np.int64), v[vis].astype(np.int64)
    m, L = mask[vi, ui], labels[vis]
    lab = L != 255
    total = np.bincount(L[lab], minlength=NUM_CLASSES)[:NUM_CLASSES]
    hits = np.bincount(L[lab & (m == L)], minlength=NUM_CLASSES)[:NUM_CLASSES]
    valid = lab & (m != 255)
    return hits, total, int((m[valid] == L[valid]).sum()), int(valid.sum())


def image_records(xyz: np.ndarray, labels: np.ndarray, mask: np.ndarray, pose: Pose, cal: Calibration, image_id: str,
                  trans_cm: Sequence[float] = TRANS_CM, rot_deg: Sequence[float] = ROT_DEG, draws: int = 1,
                  seed: int = 0, noise: str = "protocol", max_dist: float = 150.0, ds: int = 4, tau: float = 0.5) -> List[Dict]:
    """All evaluations of one image: the delivered pose once, then `draws` draws of every level of the two error
    kinds. xyz and pose.S are in the same frame (zone points and camera centre relative to the cache origin).
    Returns records {"image", "kind", "level", "draw", "hits", "total", "hits_all", "n_all"} (plain lists / ints)."""
    cand = candidates(xyz, pose, cal, max_dist=max_dist)
    pts, lab = xyz[cand].astype(np.float64), labels[cand]
    plan = [("none", 0.0, 0)] + [(kind, float(lv), d) for kind, levels in (("translation_cm", trans_cm), ("rotation_deg", rot_deg))
                                 for lv in levels for d in range(draws)]
    out = []
    for kind, level, d in plan:
        p = perturbed(pose, kind, level, draw_rng(seed, image_id, kind, level, d), noise)
        hits, total, ha, na = transfer_counts(pts, lab, mask, p, cal, ds=ds, tau=tau)
        out.append({"image": image_id, "kind": kind, "level": level, "draw": d, "hits": [int(x) for x in hits],
                    "total": [int(x) for x in total], "hits_all": ha, "n_all": na})
    return out


def _share(h, t):
    return float(h) / float(t) if t else float("nan")


def summarize(records: Iterable[Dict], min_points: int = 50, names: Sequence[str] = CLASS_NAMES) -> List[Dict]:
    """Pooled result per (kind, level): one row {"perturbation", "level", "all", <class>..., "n", "images", "draws",
    "all_q25", "all_q75", "<class>_q25", "<class>_q75"}. A share is pooled over all images and draws of the level
    (sum of hits / sum of points). q25 / q75 are the quartiles over the images of the per-image share (pooled over
    the draws of that image), over the images with at least `min_points` points of the class; NaN when no image
    qualifies. "n" is the mean number of point-pixel pairs behind "all" per draw. The delivered poses are the row
    ("none", 0)."""
    groups: Dict = {}
    for r in records:
        groups.setdefault((r["kind"], float(r["level"])), []).append(r)
    rows = []
    for (kind, level) in sorted(groups, key=lambda k: (KINDS[k[0]], k[1])):
        rs = groups[(kind, level)]
        hits = np.sum([r["hits"] for r in rs], axis=0); total = np.sum([r["total"] for r in rs], axis=0)
        ha = sum(r["hits_all"] for r in rs); na = sum(r["n_all"] for r in rs)
        per_img: Dict[str, List] = {}
        for r in rs:
            a = per_img.setdefault(r["image"], [np.zeros(len(names), np.int64), np.zeros(len(names), np.int64), 0, 0])
            a[0] += np.asarray(r["hits"]); a[1] += np.asarray(r["total"]); a[2] += r["hits_all"]; a[3] += r["n_all"]
        row = {"perturbation": kind, "level": level, "all": _share(ha, na)}
        row.update({n: _share(hits[i], total[i]) for i, n in enumerate(names)})
        row.update({"n": na / max(len({(r["image"], r["draw"]) for r in rs}), 1), "images": len(per_img),
                    "draws": max(r["draw"] for r in rs) + 1})

        def quartiles(values):
            return [float(x) for x in np.percentile(values, [25, 75])] if len(values) else [float("nan")] * 2

        row["all_q25"], row["all_q75"] = quartiles([a[2] / a[3] for a in per_img.values() if a[3] >= min_points])
        for i, n in enumerate(names):
            row[n + "_q25"], row[n + "_q75"] = quartiles([a[0][i] / a[1][i] for a in per_img.values() if a[1][i] >= min_points])
        rows.append(row)
    return rows


def csv_columns(names: Sequence[str] = CLASS_NAMES) -> List[str]:
    return (["perturbation", "level", "noise", "all"] + list(names) + ["n", "images", "draws", "all_q25", "all_q75"]
            + [f"{n}_{q}" for n in names for q in ("q25", "q75")])


def write_csv(path: str, rows: Sequence[Dict], noise: str, names: Sequence[str] = CLASS_NAMES) -> str:
    """The table the figure is drawn from (scripts/plot_label_transfer.py); shares as fractions in [0, 1]."""
    cols = csv_columns(names)
    with open(path + ".tmp", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([{**r, "noise": noise}[c] for c in cols])
    os.replace(path + ".tmp", path)
    return path
