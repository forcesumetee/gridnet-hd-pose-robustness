"""Point ↔ image association: for every 3D point keep the K best views with their pixel
coordinates and viewing-condition features. This table is the input of the image-lifting and
learned view-weighting modules (C1) and of the misalignment protocol (C2).

Per (point, view) pair we store:
  img   int16   index into `image_ids` (-1 = empty slot)
  u, v  uint16  pixel coordinates (full-resolution image)
  feat  float16 [dist_m, cos_incidence, depth_residual_m, radial_pos, depth_m]
Ranking for the K slots: nearest camera first (score = -distance).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..camera import Calibration, Pose, frustum_prefilter, project, visible

FEAT_NAMES = ["dist", "cos_inc", "depth_resid", "radial", "depth"]


@dataclass
class Association:
    image_ids: List[str]
    img: np.ndarray    # (N, K) int16
    u: np.ndarray      # (N, K) uint16
    v: np.ndarray      # (N, K) uint16
    feat: np.ndarray   # (N, K, 5) float16
    n_views: np.ndarray  # (N,) uint8

    def save(self, path: str):
        np.savez(path, image_ids=np.array(self.image_ids), img=self.img, u=self.u, v=self.v,
                 feat=self.feat, n_views=self.n_views, feat_names=np.array(FEAT_NAMES))

    @classmethod
    def load(cls, path: str) -> "Association":
        with np.load(path) as d:
            return cls(image_ids=list(d["image_ids"]), img=d["img"], u=d["u"], v=d["v"],
                       feat=d["feat"], n_views=d["n_views"])


def build_association(xyz: np.ndarray, poses: Dict[str, Pose], cal: Calibration, k: int = 8,
                      normals: Optional[np.ndarray] = None, ds: int = 4, tau: float = 0.5,
                      max_dist: float = 150.0, conv=None, verbose: bool = True) -> Association:
    n = len(xyz)
    image_ids = sorted(poses.keys())
    if len(image_ids) > 32000:
        raise ValueError("int16 image index overflow")
    score = np.full((n, k), -np.inf, dtype=np.float32)
    img = np.full((n, k), -1, dtype=np.int16)
    u16 = np.zeros((n, k), dtype=np.uint16)
    v16 = np.zeros((n, k), dtype=np.uint16)
    feat = np.zeros((n, k, len(FEAT_NAMES)), dtype=np.float16)
    n_views = np.zeros(n, dtype=np.uint8)
    cx, cy = cal.width / 2, cal.height / 2
    kwargs = {} if conv is None else {"conv": conv}

    for i, iid in enumerate(image_ids):
        pose = poses[iid]
        cand = frustum_prefilter(xyz, pose, cal, max_dist=max_dist)
        if len(cand) == 0:
            continue
        u, v, z, ok = project(xyz[cand], pose, cal, **kwargs)
        vis, resid = visible(u, v, z, ok, cal, ds=ds, tau=tau)
        sel = cand[vis]
        if len(sel) == 0:
            continue
        uu, vv, zz, rr = u[vis], v[vis], z[vis], resid[vis]
        d = xyz[sel] - pose.S
        dist = np.linalg.norm(d, axis=1)
        if normals is not None:
            cos_inc = np.abs(np.einsum("ij,ij->i", normals[sel], d / dist[:, None]))
        else:
            cos_inc = np.ones(len(sel), dtype=np.float32)
        radial = np.sqrt(((uu - cx) / cx) ** 2 + ((vv - cy) / cy) ** 2)
        new_score = (-dist).astype(np.float32)

        # replace the worst slot where the new view is better
        cur = score[sel]                              # (M, K)
        worst = np.argmin(cur, axis=1)
        better = new_score > cur[np.arange(len(sel)), worst]
        rows = sel[better]; slots = worst[better]
        score[rows, slots] = new_score[better]
        img[rows, slots] = i
        u16[rows, slots] = np.clip(uu[better], 0, 65535).astype(np.uint16)
        v16[rows, slots] = np.clip(vv[better], 0, 65535).astype(np.uint16)
        f = np.stack([dist[better], cos_inc[better], rr[better], radial[better], zz[better]], axis=1)
        feat[rows, slots] = f.astype(np.float16)
        n_views[sel] = np.minimum(n_views[sel].astype(np.int32) + 1, 255).astype(np.uint8)
        if verbose:
            print(f"\r[assoc] {i + 1}/{len(image_ids)} {iid}: visible {len(sel):,}", end="", flush=True)
    if verbose:
        print()
    return Association(image_ids=image_ids, img=img, u=u16, v=v16, feat=feat, n_views=n_views)


def coverage_report(assoc: Association, labels: np.ndarray, class_names: Dict[int, str]) -> str:
    """How many views each class gets — sanity check and a number for the thesis."""
    lines = [f"{'class':<26}{'points':>12}{'0 views':>9}{'mean views':>12}"]
    for c in sorted(set(np.unique(labels).tolist()) - {255}):
        m = labels == c
        nv = assoc.n_views[m].astype(np.float32)
        lines.append(f"{class_names.get(c, str(c)):<26}{int(m.sum()):>12,}{float((nv == 0).mean()) * 100:>8.1f}%"
                     f"{float(nv.mean()):>12.2f}")
    return "\n".join(lines)
