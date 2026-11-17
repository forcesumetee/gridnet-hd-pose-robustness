"""GPU (torch) point–image association: the same model as gridnet.preprocess.association
(Metashape frame camera, z-buffer visibility, K nearest views) but 20–50x faster, chunked over points so a
50–80 M point zone fits a 24 GB GPU. Produces the same `Association` container, and accepts any dict of
poses, so the robustness protocol can re-associate with perturbed poses.

  assoc = associate(xyz_rel, origin, poses, cal, image_ids, k=8, tau=0.5, device="cuda")
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..camera import VALIDATED_CONV, Calibration, Pose
from ..preprocess.association import FEAT_NAMES, Association


def project_t(xyz: torch.Tensor, S: torch.Tensor, R: torch.Tensor, cal: Calibration, conv=VALIDATED_CONV,
              fov_margin: float = 1.15):
    """torch port of gridnet.camera.project (xyz: (N,3) float32 world coords; S (3,), R (3,3) float32)."""
    sx, sy, dz = conv
    P = (xyz - S) @ R.T
    Zc = P[:, 2]
    z = dz * (-Zc)
    front = z > 1e-6
    safe = torch.where(front, Zc, torch.ones_like(Zc))
    x = torch.where(front, sx * (-P[:, 0] / safe), torch.zeros_like(Zc))
    y = torch.where(front, sy * (-P[:, 1] / safe), torch.zeros_like(Zc))
    tx, ty = cal.half_fov_tan
    front = front & (x.abs() <= fov_margin * tx) & (y.abs() <= fov_margin * ty)
    x = torch.where(front, x, torch.zeros_like(x)); y = torch.where(front, y, torch.zeros_like(y))
    rc = x * x + y * y
    dr = 1 + cal.k1 * rc + cal.k2 * rc ** 2 + cal.k3 * rc ** 3 + cal.k4 * rc ** 4 + cal.k5 * rc ** 5
    tg = 1 + cal.p3 * rc + cal.p4 * rc ** 2
    xp = x * dr + cal.p1 * (rc + 2 * x * x) + 2 * cal.p2 * x * y * tg
    yp = y * dr + cal.p2 * (rc + 2 * y * y) + 2 * cal.p1 * x * y * tg
    u = cal.width / 2 + cal.cx + xp * cal.f + xp * cal.b1 + yp * cal.b2
    v = cal.height / 2 + cal.cy + yp * cal.f
    ok = front & torch.isfinite(u) & torch.isfinite(v) & (u >= 0) & (u < cal.width) & (v >= 0) & (v < cal.height)
    return u, v, z, ok


def frustum_prefilter_t(xyz: torch.Tensor, S: torch.Tensor, R: torch.Tensor, cal: Calibration,
                        max_dist: float = 150.0, margin: float = 1.3) -> torch.Tensor:
    d = xyz - S
    dist = torch.linalg.norm(d, dim=1)
    zc = d @ (-R[2])
    cand = (zc > 0.5) & (dist < max_dist)
    tx, ty = cal.half_fov_tan
    half_tan = margin * float(np.hypot(tx, ty))
    lateral = torch.sqrt(torch.clamp(dist * dist - zc * zc, min=0))
    return torch.nonzero(cand & (lateral <= half_tan * zc), as_tuple=False).squeeze(1)


def depth_grid(cal: Calibration, ds: int = 4):
    return cal.width // ds + 1, cal.height // ds + 1


def depth_accumulate(depth: torch.Tensor, u, v, z, ok, W: int, ds: int = 4) -> None:
    """Scatter the minimum depth of the ok points into the flat (W*H,) depth map of one image."""
    idx = torch.nonzero(ok, as_tuple=False).squeeze(1)
    if len(idx) == 0:
        return
    lin = (v[idx] / ds).long() * W + (u[idx] / ds).long()
    depth.scatter_reduce_(0, lin, z[idx], reduce="amin", include_self=True)


def visible_t(depth: torch.Tensor, u, v, z, ok, W: int, ds: int = 4, tau: float = 0.5):
    """Visibility against a finished depth map. Returns vis (bool) and residual z - depth for all inputs."""
    idx = torch.nonzero(ok, as_tuple=False).squeeze(1)
    lin = (v[idx] / ds).long() * W + (u[idx] / ds).long()
    resid = torch.zeros_like(z)
    resid[idx] = z[idx] - depth[lin]
    vis = torch.zeros_like(ok)
    vis[idx] = resid[idx] <= tau
    return vis, resid


def zone_cams(poses: Dict[str, Pose], image_ids: Sequence[str], origin, device: str = "cuda"):
    """Camera centres (n, 3) and rotations (n, 3, 3) as float32 tensors, centres shifted by -origin so that
    everything works in zone-relative float32 coordinates."""
    o = np.asarray(origin, dtype=np.float64)
    S = torch.as_tensor(np.stack([poses[i].S - o for i in image_ids]), dtype=torch.float32, device=device)
    R = torch.as_tensor(np.stack([poses[i].R for i in image_ids]), dtype=torch.float32, device=device)
    return S, R


@torch.no_grad()
def depth_maps(xyz_rel: np.ndarray, S: torch.Tensor, R: torch.Tensor, cal: Calibration, chunk: int = 20_000_000,
               ds: int = 4, max_dist: float = 150.0, verbose: bool = True) -> Tuple[torch.Tensor, int, int]:
    """z-buffer depth map of every camera from ALL points (chunked). Returns (depths (n_img, W*H) float32, W, H)."""
    device = S.device
    W, H = depth_grid(cal, ds)
    n = len(xyz_rel)
    depths = torch.full((S.shape[0], W * H), float("inf"), device=device)
    n_chunks = (n + chunk - 1) // chunk
    for c0 in range(0, n, chunk):
        xyz = torch.as_tensor(xyz_rel[c0:min(n, c0 + chunk)], dtype=torch.float32, device=device)
        for i in range(S.shape[0]):
            cand = frustum_prefilter_t(xyz, S[i], R[i], cal, max_dist=max_dist)
            if len(cand) == 0:
                continue
            u, v, z, ok = project_t(xyz[cand], S[i], R[i], cal)
            depth_accumulate(depths[i], u, v, z, ok, W, ds)
        if verbose:
            print(f"\r[assoc] depth maps: chunk {c0 // chunk + 1}/{n_chunks}", end="", flush=True)
        del xyz
    return depths, W, H


@torch.no_grad()
def associate(xyz_rel: np.ndarray, origin: np.ndarray, poses: Dict[str, Pose], cal: Calibration,
              image_ids: Optional[Sequence[str]] = None, k: int = 8, tau: float = 0.5, ds: int = 4,
              max_dist: float = 150.0, normals: Optional[np.ndarray] = None, chunk: int = 20_000_000,
              device: str = "cuda", verbose: bool = True, depths: Optional[torch.Tensor] = None) -> Association:
    """K nearest visible views per point. xyz_rel are cache coordinates (float32, relative to `origin`).
    `depths` (from `depth_maps`) can be passed to skip pass 1."""
    image_ids = sorted(image_ids if image_ids is not None else poses.keys())
    assert len(image_ids) < 32000
    n = len(xyz_rel)
    cx, cy = cal.width / 2, cal.height / 2
    out_img = np.full((n, k), -1, np.int16); out_u = np.zeros((n, k), np.uint16); out_v = np.zeros((n, k), np.uint16)
    out_feat = np.zeros((n, k, len(FEAT_NAMES)), np.float16); out_nv = np.zeros(n, np.uint8)
    S_all, R_all = zone_cams(poses, image_ids, origin, device)
    cams = [(S_all[i], R_all[i]) for i in range(len(image_ids))]
    # Pass 1: z-buffer depth map of every image from ALL points (chunked), so occlusion is complete.
    W, H = depth_grid(cal, ds)
    if depths is None:
        depths, W, H = depth_maps(xyz_rel, S_all, R_all, cal, chunk=chunk, ds=ds, max_dist=max_dist, verbose=verbose)
    n_chunks = (n + chunk - 1) // chunk
    # Pass 2: project again, test visibility against the finished depth maps, keep the K nearest views.
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        xyz = torch.as_tensor(xyz_rel[c0:c1], dtype=torch.float32, device=device)
        nrm = None if normals is None else torch.as_tensor(np.asarray(normals[c0:c1], np.float32), device=device)
        m = c1 - c0
        score = torch.full((m, k), -float("inf"), device=device)
        img = torch.full((m, k), -1, dtype=torch.int16, device=device)
        uu16 = torch.zeros((m, k), dtype=torch.int16, device=device); vv16 = torch.zeros((m, k), dtype=torch.int16, device=device)
        feat = torch.zeros((m, k, len(FEAT_NAMES)), dtype=torch.float16, device=device)
        nv = torch.zeros(m, dtype=torch.int16, device=device)
        for i, (S, R) in enumerate(cams):
            cand = frustum_prefilter_t(xyz, S, R, cal, max_dist=max_dist)
            if len(cand) == 0:
                continue
            u, v, z, ok = project_t(xyz[cand], S, R, cal)
            vis, resid = visible_t(depths[i], u, v, z, ok, W, ds=ds, tau=tau)
            sel_local = torch.nonzero(vis, as_tuple=False).squeeze(1)
            if len(sel_local) == 0:
                continue
            sel = cand[sel_local]
            u, v, z, resid = u[sel_local], v[sel_local], z[sel_local], resid[sel_local]
            d = xyz[sel] - S
            dist = torch.linalg.norm(d, dim=1)
            cos_inc = (nrm[sel] * (d / dist[:, None])).sum(1).abs() if nrm is not None else torch.ones_like(dist)
            radial = torch.sqrt(((u - cx) / cx) ** 2 + ((v - cy) / cy) ** 2)
            new_score = -dist
            cur = score[sel]                                   # (M, K)
            worst = cur.argmin(1)
            better = new_score > cur.gather(1, worst[:, None]).squeeze(1)
            rows, slots = sel[better], worst[better]
            score[rows, slots] = new_score[better]
            img[rows, slots] = i
            uu16[rows, slots] = u[better].clamp(0, 32767).to(torch.int16)
            vv16[rows, slots] = v[better].clamp(0, 32767).to(torch.int16)
            feat[rows, slots] = torch.stack([dist[better], cos_inc[better], resid[better], radial[better], z[better]], 1).half()
            nv[sel] = torch.clamp(nv[sel] + 1, max=255)
            if verbose and (i + 1) % 25 == 0:
                print(f"\r[assoc] views: chunk {c0 // chunk + 1}/{n_chunks}  image {i + 1}/{len(cams)}", end="", flush=True)
        out_img[c0:c1] = img.cpu().numpy(); out_u[c0:c1] = uu16.cpu().numpy().astype(np.uint16)
        out_v[c0:c1] = vv16.cpu().numpy().astype(np.uint16); out_feat[c0:c1] = feat.cpu().numpy()
        out_nv[c0:c1] = nv.clamp(max=255).cpu().numpy().astype(np.uint8)
        del xyz, score, img, uu16, vv16, feat, nv, nrm
        if device == "cuda":
            torch.cuda.empty_cache()
    if verbose:
        print()
    return Association(image_ids=list(image_ids), img=out_img, u=out_u, v=out_v, feat=out_feat, n_views=out_nv)
