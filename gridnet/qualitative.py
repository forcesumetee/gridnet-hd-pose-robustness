"""Crops of protocol predictions around pylons, for the qualitative figure of the paper (numpy only).

The robustness protocol predicts a class-stratified subset of every validation zone: the points of the four
electrical classes (all of them up to the cap of the subset) and a sample of every other class. `CropDump` is a hook
for `gridnet.train.protocol.run_protocol(on_draw=...)`: for the first draw of chosen error levels it keeps the
predictions of chosen models for the subset points within `radius` (horizontal distance) of the largest pylons of the
zone, and writes one small file per pylon:

    <out_dir>/<zone>_crop<i>.npz
        xyz_cm (n,3) int16    centimetres relative to the pylon centre (x, y) and to the lowest point of the crop (z);
                              the kept points are in spatial order (x, then y, then z), not in file order
        labels (n,)  uint8    ground truth of the kept points
        preds  (K,n) uint8    predicted class per kept point, one row per key
        keys   (K,)  str      "model|sweep|level"
        cm     (K,11,11)      weighted confusion matrix over ALL subset points of the crop (rows = ground truth)
        class_points (11,)    weighted number of points per class in the crop
        centre (2,), z0, radius, voxel, n_subset, zone

Two things are separated on purpose. The STATISTICS of a crop (`cm`) use every subset point within the radius with
its sampling weight, so they are the numbers of the benchmark restricted to that place. The POINTS are thinned for
drawing: one point per `voxel` (the first in file order), and when more than `max_points` remain the surroundings and
then the pylon are subsampled further (the thin classes stay) - a figure at 10 cm per pixel does not show more, and
the files stay small enough to be fetched from Drive. The surroundings are sparse in any case (they are a sample).

`view_coords` gives the coordinates of an orthographic view of a crop; `scripts/plot_qualitative.py` draws the panels.
"""
import os
import zlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ELECTRICAL = (0, 1, 2, 3)            # pylon, conductor cable, structural cable, insulator
THIN = (1, 2, 3)
PYLON, CONDUCTOR = 0, 1
NUM = 11


def _numpy(x) -> np.ndarray:
    """A torch tensor (any device) or an array -> numpy array, without importing torch here."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def pylon_centres(xyz: np.ndarray, labels: np.ndarray, n: int = 2, bin_m: float = 10.0, min_points: int = 500,
                  min_sep: float = 40.0) -> List[np.ndarray]:
    """Up to `n` pylon centres (x, y), largest first: peaks of a coarse 2D histogram of the pylon points that hold at
    least `min_points` points in their 3 x 3 bins and lie at least `min_sep` metres apart. Without such a peak the
    result is one centre: the median of the electrical points, or of all points when the zone has none."""
    p = xyz[labels == PYLON, :2].astype(np.float64)
    out: List[np.ndarray] = []
    if len(p) >= min_points:
        lo = p.min(0)
        ij = np.floor((p - lo) / bin_m).astype(np.int64)
        nx, ny = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
        h = np.bincount(ij[:, 0] * ny + ij[:, 1], minlength=nx * ny).reshape(nx, ny)
        for flat in np.argsort(h, axis=None)[::-1]:
            if len(out) >= n or h.flat[flat] == 0:
                break
            i, j = np.unravel_index(flat, h.shape)
            near = (np.abs(ij[:, 0] - i) <= 1) & (np.abs(ij[:, 1] - j) <= 1)
            if near.sum() < min_points:
                continue
            c = p[near].mean(0)
            if all(np.linalg.norm(c - o) >= min_sep for o in out):
                out.append(c)
    if not out:
        e = xyz[np.isin(labels, ELECTRICAL), :2]
        src = e if len(e) else xyz[:, :2]
        out = [np.median(src.astype(np.float64), axis=0)]
    return out


def crop_indices(xyz: np.ndarray, centre: Sequence[float], radius: float) -> np.ndarray:
    """Indices of the points within `radius` (horizontal distance) of `centre`."""
    d2 = ((xyz[:, :2].astype(np.float64) - np.asarray(centre, np.float64)) ** 2).sum(1)
    return np.flatnonzero(d2 <= radius ** 2)


def thin_indices(rel: np.ndarray, labels: np.ndarray, voxel: float, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices of the points kept for drawing: the first point of every `voxel` cell (voxel <= 0: all). When
    more than `max_points` remain, the budget goes to the thin classes first (conductor, structural cable, insulator:
    all of them, up to four fifths of the budget), then to the surroundings (what is left once the pylon points are
    counted, at least a fifth of the budget) and the rest to the pylon points; each group is subsampled uniformly."""
    keep = np.arange(len(rel))
    if voxel > 0 and len(rel):
        q = np.floor((rel - rel.min(0)) / voxel).astype(np.int64)
        ny, nz = int(q[:, 1].max()) + 1, int(q[:, 2].max()) + 1
        _, first = np.unique((q[:, 0] * ny + q[:, 1]) * nz + q[:, 2], return_index=True)
        keep = np.sort(first)
    if len(keep) > max_points > 0:
        lab = labels[keep]
        thin, pyl, back = keep[np.isin(lab, THIN)], keep[lab == PYLON], keep[~np.isin(lab, ELECTRICAL)]
        n_thin = min(len(thin), max_points - max_points // 5)
        n_back = min(len(back), max(max_points // 5, max_points - n_thin - len(pyl)))
        n_pyl = min(len(pyl), max_points - n_thin - n_back)
        parts = [g if len(g) <= n else rng.choice(g, n, replace=False) for g, n in ((thin, n_thin), (pyl, n_pyl), (back, n_back))]
        keep = np.sort(np.concatenate(parts))
    return keep


def spatial_order(q: np.ndarray) -> np.ndarray:
    """Order of the kept points in a crop file: by x, then y, then z of the stored (integer) coordinates. Neighbours
    in the file are neighbours in space, which roughly halves the size of the compressed file."""
    return np.lexsort((q[:, 2], q[:, 1], q[:, 0]))


def weighted_cm(gt: np.ndarray, pred: np.ndarray, w: np.ndarray, n: int = NUM) -> np.ndarray:
    """(n, n) confusion matrix with point weights; rows = ground truth, columns = prediction."""
    return np.bincount(gt.astype(np.int64) * n + pred.astype(np.int64), weights=w.astype(np.float64), minlength=n * n).reshape(n, n)


def wrong_share(cm: np.ndarray, classes: Sequence[int] = ELECTRICAL) -> float:
    """Share of the points whose ground truth is in `classes` that got another class (nan without such points)."""
    cm = np.asarray(cm, np.float64)
    rows = list(classes)
    total = cm[rows].sum()
    return float("nan") if total <= 0 else float(1.0 - sum(cm[c, c] for c in rows) / total)


def parse_levels(text: str) -> List[Tuple[str, float]]:
    """"trans:0,trans:0.5,rot:1" -> [("trans", 0.0), ("trans", 0.5), ("rot", 1.0)]"""
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            sweep, level = item.split(":")
            out.append((sweep.strip(), float(level)))
    return out


def key(model: str, sweep: str, level: float) -> str:
    return f"{model}|{sweep}|{float(level):g}"


def split_key(k: str) -> Tuple[str, str, float]:
    model, sweep, level = k.rsplit("|", 2)
    return model, sweep, float(level)


class CropDump:
    """The `on_draw` hook of run_protocol: keeps draw 0 of the wanted (sweep, level) pairs for the wanted models."""

    def __init__(self, out_dir: str, levels: Iterable[Tuple[str, float]], models: Optional[Sequence[str]] = None,
                 radius: float = 35.0, pylons: int = 2, min_points: int = 500, voxel: float = 0.05,
                 max_points: int = 300_000, seed: int = 0):
        self.out_dir, self.radius, self.pylons, self.min_points = out_dir, float(radius), int(pylons), int(min_points)
        self.voxel, self.max_points, self.seed = float(voxel), int(max_points), int(seed)
        self.levels = {(s, round(float(lv), 6)) for s, lv in levels}
        self.models = None if not models else list(models)
        self.zones: Dict[str, list] = {}
        os.makedirs(out_dir, exist_ok=True)

    def settings(self) -> dict:
        return {"levels": sorted([s, lv] for s, lv in self.levels), "models": self.models, "radius": self.radius, "pylons": self.pylons,
                "min_points": self.min_points, "voxel": self.voxel, "max_points": self.max_points, "seed": self.seed}

    def _crops(self, zone: str, xyz, labels, weights) -> list:
        if zone not in self.zones:
            xyz_np, lab_np = _numpy(xyz).astype(np.float32), _numpy(labels).astype(np.uint8)
            w_np = np.ones(len(lab_np), np.float64) if weights is None else _numpy(weights).astype(np.float64)
            rng = np.random.default_rng(self.seed + zlib.crc32(zone.encode()))
            crops = []
            for c in pylon_centres(xyz_np, lab_np, n=self.pylons, min_points=self.min_points):
                idx = crop_indices(xyz_np, c, self.radius)
                if len(idx) == 0:
                    continue
                rel = xyz_np[idx].astype(np.float64)
                z0 = float(rel[:, 2].min())
                rel[:, :2] -= c
                rel[:, 2] -= z0
                keep = thin_indices(rel, lab_np[idx], self.voxel, self.max_points, rng)
                q = np.round(rel[keep] * 100.0).astype(np.int64)
                order = spatial_order(q)
                keep, q = keep[order], q[order]
                crops.append({"centre": np.asarray(c, np.float64), "z0": z0, "idx": idx, "keep": keep, "gt": lab_np[idx], "w": w_np[idx],
                              "xyz_cm": np.asfortranarray(q.astype(np.int16 if np.abs(q).max() < 32767 else np.int32)), "preds": {}, "cm": {}})
            self.zones[zone] = crops
        return self.zones[zone]

    def __call__(self, zone: str, sweep: str, level: float, draw: int, preds: dict, xyz, labels, weights=None, **_) -> None:
        if draw != 0 or (sweep, round(float(level), 6)) not in self.levels:
            return
        crops = self._crops(zone, xyz, labels, weights)
        for name, p in preds.items():
            if self.models is None or name in self.models:
                p_np = _numpy(p).astype(np.uint8)
                for crop in crops:
                    sub = p_np[crop["idx"]]
                    k = key(name, sweep, level)
                    crop["preds"][k] = sub[crop["keep"]]
                    crop["cm"][k] = weighted_cm(crop["gt"], sub, crop["w"]).astype(np.float32)
        self.save(zone)

    def paths(self, zone: str) -> List[str]:
        return [os.path.join(self.out_dir, f"{zone}_crop{i}.npz") for i in range(len(self.zones.get(zone, [])))]

    def save(self, zone: str) -> List[str]:
        """(Re)write the files of the zone; called after every kept draw, so an interrupted run leaves what it had."""
        for path, crop in zip(self.paths(zone), self.zones[zone]):
            keys = sorted(crop["preds"])
            n = len(crop["keep"])
            preds = np.stack([crop["preds"][k] for k in keys]) if keys else np.zeros((0, n), np.uint8)
            cm = np.stack([crop["cm"][k] for k in keys]) if keys else np.zeros((0, NUM, NUM), np.float32)
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, xyz_cm=crop["xyz_cm"], labels=crop["gt"][crop["keep"]], preds=preds, keys=np.array(keys, dtype=str), cm=cm,
                                class_points=np.bincount(crop["gt"], weights=crop["w"], minlength=NUM)[:NUM], centre=crop["centre"],
                                z0=np.float64(crop["z0"]), radius=np.float64(self.radius), voxel=np.float64(self.voxel),
                                n_subset=np.int64(len(crop["idx"])), zone=np.array(zone))
            os.replace(tmp, path)
        return self.paths(zone)


def load_crop(path: str) -> dict:
    """A file written by CropDump: {"xyz" (n,3) float32 metres, "labels", "preds": {key: (n,) uint8},
    "cm": {key: (11,11)}, "class_points", "centre", "z0", "radius", "voxel", "n_subset", "zone"}."""
    with np.load(path) as f:
        keys = [str(k) for k in f["keys"]]
        preds, cm = f["preds"], f["cm"]
        return {"xyz": f["xyz_cm"].astype(np.float32) / 100.0, "labels": f["labels"],
                "preds": {k: preds[i] for i, k in enumerate(keys)}, "cm": {k: cm[i] for i, k in enumerate(keys)},
                "class_points": f["class_points"], "centre": f["centre"], "z0": float(f["z0"]), "radius": float(f["radius"]),
                "voxel": float(f["voxel"]), "n_subset": int(f["n_subset"]), "zone": str(f["zone"])}


def line_direction(xyz: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Unit horizontal vector along the power line: the main axis of the conductor points (of the power-line points,
    then of all points, when there are fewer than 100 of them). Its larger component is positive."""
    for classes in ((CONDUCTOR,), ELECTRICAL, None):
        src = xyz[:, :2] if classes is None else xyz[np.isin(labels, classes), :2]
        if classes is None or len(src) >= 100:
            break
    src = src.astype(np.float64)
    if len(src) < 2:
        return np.array([1.0, 0.0])
    _, _, vt = np.linalg.svd(src - src.mean(0), full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])
    return -d if d[np.argmax(np.abs(d))] < 0 else d


def view_axes(direction: Sequence[float], azimuth: float = 90.0, elevation: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, towards the viewer) unit vectors of an orthographic view. The viewer stands `azimuth` degrees from
    the line direction (90 = looks across the line, 0 = along it) and `elevation` degrees above the horizon."""
    d = np.asarray(direction, np.float64)
    n = np.array([-d[1], d[0]])
    a, e = np.radians(azimuth), np.radians(elevation)
    h = np.cos(a) * d + np.sin(a) * n                      # horizontal direction from the scene to the viewer
    v = np.array([np.cos(e) * h[0], np.cos(e) * h[1], np.sin(e)])
    right = np.array([-h[1], h[0], 0.0])
    return right, np.cross(v, right), v


def view_coords(xyz: np.ndarray, labels: np.ndarray, azimuth: float = 90.0, elevation: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, depth) of every point of a crop: u to the viewer's right, v up, depth towards the viewer (a z-buffer
    keeps the largest). With the default angles u runs along the conductors and v is the height."""
    right, up, towards = view_axes(line_direction(xyz, labels), azimuth, elevation)
    p = xyz.astype(np.float64)
    return p @ right, p @ up, p @ towards
