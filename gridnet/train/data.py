"""Training/evaluation data for cached GridNet-HD zones (numpy side; torch conversion at the end).

A "zone store" holds, per zone, xyz (float32, zone-relative), labels (uint8, 11-class, 255 ignore)
and the per-point features used as network input. Crops are cylinders in xy (radius r) with a
point cap; evaluation walks a regular grid of overlapping cylinders so every point is predicted at
least once and votes are averaged.
"""
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


class GridIndex:
    """Memory-light replacement for cKDTree.query_ball_point on xy: points are bucketed into
    square cells of size `cell`; a radius query gathers the cells touching the query box and
    filters exactly. Costs 4 bytes/point (sorted index) instead of a full tree."""

    def __init__(self, xy: np.ndarray, cell: float = 5.0):
        self.cell = cell
        self.lo = xy.min(0)
        ij = np.floor((xy - self.lo) / cell).astype(np.int64)
        self.nx = int(ij[:, 0].max()) + 1
        self.ny = int(ij[:, 1].max()) + 1
        key = ij[:, 0] * self.ny + ij[:, 1]
        self.order = np.argsort(key, kind="stable").astype(np.int32)
        counts = np.bincount(key, minlength=self.nx * self.ny)
        self.starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.xy = xy

    def query_ball_point(self, c, r: float) -> np.ndarray:
        c = np.asarray(c, dtype=np.float64)
        i0, j0 = np.floor((c - r - self.lo) / self.cell).astype(np.int64)
        i1, j1 = np.floor((c + r - self.lo) / self.cell).astype(np.int64)
        i0, i1 = max(i0, 0), min(i1, self.nx - 1)
        j0, j1 = max(j0, 0), min(j1, self.ny - 1)
        if i0 > i1 or j0 > j1:
            return np.zeros(0, np.int64)
        parts = []
        for i in range(i0, i1 + 1):
            a, b = self.starts[i * self.ny + j0], self.starts[i * self.ny + j1 + 1]
            if b > a:
                parts.append(self.order[a:b])
        if not parts:
            return np.zeros(0, np.int64)
        cand = np.concatenate(parts).astype(np.int64)
        d2 = ((self.xy[cand] - c) ** 2).sum(1)
        return cand[d2 <= r * r]

FEATURE_SETS = {
    "xyz": [],                                                   # coordinates only (added always)
    "geo": ["linearity", "planarity", "verticality", "hag"],      # default
    "geo+": ["linearity", "planarity", "sphericity", "verticality", "hag"],
}
RARE = (0, 1, 2, 3)   # pylon, conductor, structural cable, insulator


@dataclass
class Zone:
    name: str
    xyz: np.ndarray        # (N, 3) float32
    labels: np.ndarray     # (N,) uint8
    feats: np.ndarray      # (N, F) float16
    tree: GridIndex        # on xy
    rare_idx: np.ndarray   # indices of rare-class points (crop centres)

    @property
    def n(self) -> int:
        return len(self.labels)


def load_zone(cache_dir: str, name: str, feature_set: str = "geo") -> Zone:
    p = np.load(os.path.join(cache_dir, name, "points.npz"))
    xyz = p["xyz"].astype(np.float32)
    labels = p["labels"].astype(np.uint8)
    names = FEATURE_SETS[feature_set]
    feats = np.stack([p[k].astype(np.float16) for k in names], axis=1) if names else np.zeros((len(xyz), 0), np.float16)
    rare = np.where(np.isin(labels, RARE))[0]
    return Zone(name=name, xyz=xyz, labels=labels, feats=feats, tree=GridIndex(xyz[:, :2]), rare_idx=rare)


def load_zones(cache_dir: str, names: Sequence[str], feature_set: str = "geo", verbose=True) -> List[Zone]:
    zones = []
    for n in names:
        zones.append(load_zone(cache_dir, n, feature_set))
        if verbose:
            print(f"[data] {n}: {zones[-1].n:,} points", flush=True)
    return zones


def class_weights(zones: Sequence[Zone], power: float = 0.5, ignore: int = 255) -> np.ndarray:
    cnt = np.zeros(11, np.float64)
    for z in zones:
        m = z.labels != ignore
        cnt += np.bincount(z.labels[m], minlength=11)
    present = cnt > 0
    w = np.ones(11, np.float64)
    w[present] = (cnt[present].sum() / cnt[present]) ** power
    w[present] /= w[present].mean()
    return w.astype(np.float32)


def cap_nearest(xy: np.ndarray, idx: np.ndarray, centre_xy, max_points: int) -> np.ndarray:
    """Cap a crop to `max_points` by keeping the points nearest (in xy) to the centre, i.e. by
    shrinking the cylinder — never by random subsampling, which would make the network see a
    sparser point density in training than at test time (Pointcept's SphereCrop does the same)."""
    if len(idx) <= max_points:
        return idx
    d2 = ((xy[idx] - np.asarray(centre_xy, dtype=xy.dtype)) ** 2).sum(1)
    return idx[np.argpartition(d2, max_points - 1)[:max_points]]


class CropSampler:
    """Random cylindrical crops for training. With probability `rare_prob` the centre is a
    rare-class point (pylon / cable / insulator), otherwise a uniformly random point.
    Zones are sampled proportionally to their point count. Crops above `max_points` are
    capped by shrinking the radius around the centre (see `cap_nearest`)."""

    def __init__(self, zones: Sequence[Zone], radius: float = 25.0, max_points: int = 600_000,
                 rare_prob: float = 0.5, seed: int = 0):
        self.zones = list(zones)
        self.radius, self.max_points, self.rare_prob = radius, max_points, rare_prob
        self.rng = np.random.default_rng(seed)
        w = np.array([z.n for z in self.zones], np.float64)
        self.zone_p = w / w.sum() if len(w) else w     # empty in eval-only mode (sampler unused)

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state(self, st: dict):
        self.rng.bit_generator.state = st["rng"]

    def sample(self) -> Tuple[Zone, np.ndarray]:
        z = self.zones[self.rng.choice(len(self.zones), p=self.zone_p)]
        if len(z.rare_idx) and self.rng.random() < self.rare_prob:
            c = z.rare_idx[self.rng.integers(len(z.rare_idx))]
        else:
            c = self.rng.integers(z.n)
        idx = np.asarray(z.tree.query_ball_point(z.xyz[c, :2], self.radius), dtype=np.int64)
        return z, cap_nearest(z.xyz[:, :2], idx, z.xyz[c, :2], self.max_points)


def augment(coord: np.ndarray, feats: np.ndarray, rng: np.random.Generator,
            rot_z=True, scale=(0.9, 1.1), flip=True, jitter=0.005) -> Tuple[np.ndarray, np.ndarray]:
    """Standard outdoor augmentation on zone-relative, crop-centred coordinates."""
    c = coord
    if rot_z:
        th = rng.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
        c = c @ R.T
    if scale is not None:
        c = c * rng.uniform(*scale)
    if flip and rng.random() < 0.5:
        c = c * np.array([-1, 1, 1], np.float32)
    if jitter:
        c = c + rng.normal(0, jitter, c.shape).astype(np.float32)
    return c.astype(np.float32), feats


def make_batch(items: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], grid: float):
    """items: list of (coord, feats, labels) per crop -> numpy dict ready for torch conversion.
    feat = [coord_centred, feats]; grid_coord from the batch-wide minimum (per-crop would also work
    since LitePT serialises per batch index)."""
    coords, feats, segs, offsets, n = [], [], [], [], 0
    for c, f, l in items:
        c = c - c.mean(0, keepdims=True)
        coords.append(c); feats.append(np.concatenate([c, f.astype(np.float32)], 1)); segs.append(l)
        n += len(c); offsets.append(n)
    coord = np.concatenate(coords).astype(np.float32)
    grid_coord = np.floor((coord - coord.min(0, keepdims=True)) / grid).astype(np.int32)
    return {"coord": coord, "grid_coord": grid_coord, "feat": np.concatenate(feats).astype(np.float32),
            "segment": np.concatenate(segs).astype(np.int64), "offset": np.array(offsets, np.int64)}


def eval_windows(zone: Zone, radius: float = 25.0, stride: Optional[float] = None,
                 max_points: int = 900_000, rng: Optional[np.random.Generator] = None) -> Iterator[np.ndarray]:
    """Regular xy grid of cylinder centres (stride = radius by default, so neighbours overlap by
    ~2x) over the zone's extent. Yields point-index arrays; windows above max_points are shrunk
    around their centre (`cap_nearest`), so dense areas may leave points uncovered — the caller
    fills those with `cover_remaining`."""
    stride = stride or radius
    xy = zone.xyz[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    xs = np.arange(lo[0], hi[0] + stride, stride)
    ys = np.arange(lo[1], hi[1] + stride, stride)
    for x in xs:
        for y in ys:
            idx = np.asarray(zone.tree.query_ball_point([x, y], radius), dtype=np.int64)
            if len(idx) == 0:
                continue
            yield cap_nearest(xy, idx, (x, y), max_points)


def cover_remaining(zone: Zone, counts: np.ndarray, radius: float, max_points: int) -> Iterator[np.ndarray]:
    """Extra windows centred on still-unpredicted points until every point has >= 1 vote.
    The caller must add to `counts[idx]` after each yielded window (as it does for the regular
    windows); each window contains its centre (distance 0), so progress is guaranteed."""
    xy = zone.xyz[:, :2]
    while True:
        rem = np.flatnonzero(counts == 0)
        if len(rem) == 0:
            return
        c = xy[rem[0]]
        idx = np.asarray(zone.tree.query_ball_point(c, radius), dtype=np.int64)
        yield cap_nearest(xy, idx, c, max_points)


class ConfusionMeter:
    def __init__(self, n=11):
        self.m = np.zeros((n, n), np.int64)

    def add(self, gt: np.ndarray, pred: np.ndarray, ignore: int = 255):
        k = gt != ignore
        np.add.at(self.m, (gt[k].astype(np.int64), pred[k].astype(np.int64)), 1)

    def iou(self) -> np.ndarray:
        inter = np.diag(self.m); union = self.m.sum(0) + self.m.sum(1) - inter
        return np.where(union > 0, inter / np.maximum(union, 1), np.nan)

    def miou(self, present_only: bool = False) -> float:
        """Mean IoU over classes with a non-empty union; with present_only, only over classes that occur in the
        ground truth (used for per-zone numbers so that a false positive of an absent class does not add a 0)."""
        iou = self.iou()
        if present_only:
            iou = np.where(self.m.sum(1) > 0, iou, np.nan)
        return float(np.nanmean(iou))

    def oa(self) -> float:
        return float(np.diag(self.m).sum() / max(self.m.sum(), 1))
