"""Robustness protocol (paper Sec. 5): how does the accuracy of a model that reads image evidence through camera
poses change when the poses are wrong by a known amount?

Step 1  pose perturbation (`draws` random draws per level; the drift has no randomness but its direction, so it is
        evaluated once per direction):
          trans  i.i.d. per image, dC ~ N(0, sigma_t^2 I)          sigma_t in SWEEPS["trans"] (m)
          rot    i.i.d. per image, R <- R(delta) R, delta ~ N(0, sigma_r^2 I)   sigma_r in SWEEPS["rot"] (deg)
          drift  systematic: images in capture order, position error along the track and heading error about the
                 vertical both grow linearly with the along-track distance to (level m, DRIFT_DEG[level] deg) at the
                 end; draw 0 = positive direction, draw 1 = negative direction
          trans_shared / rot_shared  (update41) the SAME error for every image of a zone, one per draw: dC or delta is
                 drawn once from the distribution of the i.i.d. sweep and applied to all cameras. A shared translation
                 is an offset between the image block and the point cloud (two data sets georeferenced separately),
                 a shared rotation about the camera axes is a boresight error. Same levels and draws as trans / rot:
                 the two models differ only in whether the cameras share the error. The error of a zone is seeded
                 by the zone name as well, so the zones of a draw do not get the same one.
Step 2  re-association under the perturbed poses, 2D maps and network weights fixed:
          mode "fast"   the stored K views of every point are re-projected; view set and visibility of the clean
                        association are kept, pairs that leave the image are dropped       - seconds per draw
          mode "fast_retest"  as fast, plus a visibility test of the new pixel against the CLEAN depth map (first
                        design; drops the pairs that would read a nearer object by mistake -> optimistic)
          mode "exact"  depth maps re-rendered with the perturbed cameras and the association rebuilt
                        (projection, visibility, view selection) for the evaluated points  - ~6 s per zone-draw
        Measured on two validation zones (run fast_v1, 2026-11-13, A100): fast = exact within 0.13 mIoU for every
        fusion head (0.5 IoU for any class) but up to 0.8 mIoU optimistic for ImageVote (structural cable 4.8 IoU at
        10 cm); fast_retest is off by up to 1.4 mIoU for the heads. Headline numbers: exact.
Step 3  inference + metrics on a fixed, class-stratified subset of every validation zone with the frozen LiDAR
        backbone's features/logits precomputed ONCE per zone (`precompute_subset`), so a draw costs one head forward.
        Points are weighted by the inverse of their class sampling rate, so the weighted confusion matrix is an
        unbiased estimate of the full-zone confusion matrix. Models: LiDAR-only reference (constant), ImageVote
        (uniform view mean of the 2D probabilities, fallback class for points without a view) and any number of
        trained heads (late / concat / window) read from their checkpoints.

Outputs (`run_protocol`): results.json (+ a gzip copy) with one weighted confusion matrix per (model, sweep, level,
draw, zone) and, per (zone, sweep, level, draw), how far the stored pairs moved in the image (`pixel_shift`);
`summarize` turns it into IoU-vs-error curves (mean/std over draws, pooled over zones), the robustness score
R_c = (1/sigma_max) * integral IoU_c(sigma) / IoU_c(0) (trapezoid over the levels), the drop at the reference level,
the pixel displacement per level and one pooled confusion matrix per (model, sweep, level) (confusion.json).
"""
import gzip
import json
import os
import time
import warnings
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..camera import Pose
from ..preprocess.association import Association
from .assoc_gpu import associate, depth_maps
from .data import cover_remaining, eval_windows, make_batch
from .engine import CLASS_NAMES, to_device
from .engine_fusion import fixed_torch_seed
from .fusion import DESC_SCALE, assoc_rows, build_head, lift, perturb_cams, reproject_rows, uniform_weights

SWEEPS: Dict[str, List[float]] = {
    "trans": [0.0, 0.02, 0.05, 0.10, 0.25, 0.50],     # sigma_t (m), i.i.d. per image
    "rot": [0.0, 0.05, 0.10, 0.25, 0.50, 1.0],        # sigma_r (deg), i.i.d. per image
    "drift": [0.0, 0.10, 0.25, 0.50],                 # position error (m) at the end of the flight line
    "trans_shared": [0.0, 0.02, 0.05, 0.10, 0.25, 0.50],   # sigma_t (m), ONE translation for all images of a zone
    "rot_shared": [0.0, 0.05, 0.10, 0.25, 0.50, 1.0],      # sigma_r (deg), ONE rotation for all images of a zone
}
SHARED_SWEEPS = {"trans_shared": "trans", "rot_shared": "rot"}   # shared-error sweep -> the i.i.d. sweep it is compared with
DRIFT_DEG = {0.0: 0.0, 0.10: 0.10, 0.25: 0.25, 0.50: 0.50}   # heading error (deg) paired with each drift level
REFERENCE_LEVEL = {"trans": 0.10, "rot": 0.10, "drift": 0.25,  # "drop at the reference level" of the paper
                   "trans_shared": 0.10, "rot_shared": 0.10}
ELECTRICAL = (0, 1, 2, 3)                                      # pylon, conductor, structural cable, insulator
IMAGEVOTE_FALLBACK = 6                                         # herbaceous, as in scripts/eval_imagevote.py
PROTOCOL_VERSION = 2          # 2 (update32): the drift is evaluated once per direction (draw 0 = +, draw 1 = -);
                              # 1: `draws` drift draws with a random direction each (run fast_v1)


# ----------------------------------------------------------------------------- subset + precomputed features
def stratified_subset(labels: np.ndarray, rng: np.random.Generator, cap: int = 500_000, cap_full: int = 2_000_000,
                      full_classes: Sequence[int] = ELECTRICAL, ignore: int = 255) -> Tuple[np.ndarray, np.ndarray]:
    """Indices (sorted) and inverse-sampling-rate weights of a class-stratified subset: at most `cap` points per
    class, `cap_full` for the (thin, rare) classes in `full_classes`. Unlabelled points are never selected."""
    idx_parts, w_parts = [], []
    for c in range(11):
        pts = np.flatnonzero(labels == c)
        if len(pts) == 0:
            continue
        lim = cap_full if c in full_classes else cap
        if len(pts) > lim:
            take = rng.choice(pts, lim, replace=False); rate = lim / len(pts)
        else:
            take = pts; rate = 1.0
        idx_parts.append(take); w_parts.append(np.full(len(take), 1.0 / rate, np.float32))
    idx = np.concatenate(idx_parts); w = np.concatenate(w_parts)
    order = np.argsort(idx, kind="stable")
    return idx[order], w[order]


def precompute_subset(backbone, ctx, cfg: dict, rng: np.random.Generator, cap: int = 500_000, cap_full: int = 2_000_000,
                      radius: Optional[float] = None, max_points: Optional[int] = None, verbose: bool = True) -> dict:
    """Frozen-backbone features of a stratified subset of ctx.zone, averaged over the evaluation windows that
    contain each point (same windows as FusionTrainer.predict_zone), plus the clean association rows of the
    subset. Returns a dict of numpy arrays that `save_subset` writes to disk."""
    z = ctx.zone; dev = backbone.device
    radius = radius or cfg.get("radius", 25.0); max_points = max_points or cfg.get("eval_max_points", 900_000)
    idx, w = stratified_subset(z.labels, rng, cap=cap, cap_full=cap_full)
    M = len(idx)
    pos = np.full(z.n, -1, np.int64); pos[idx] = np.arange(M)
    feat_sum = torch.zeros((M, 72), device=dev); logit_sum = torch.zeros((M, 11), device=dev)
    cnt = torch.zeros(M, device=dev)
    counts = np.zeros(z.n, np.int32)
    t0 = time.time(); n_win = 0

    def run(win):
        nonlocal n_win
        sel = pos[win]; m = sel >= 0
        counts[win] += 1; n_win += 1
        if not m.any():
            return
        batch = to_device(make_batch([(z.xyz[win], z.feats[win], z.labels[win])], cfg.get("grid", 0.05)), dev)
        feat, logits = backbone(batch)
        tgt = torch.from_numpy(sel[m]).to(dev); src = torch.from_numpy(np.flatnonzero(m)).to(dev)
        feat_sum.index_add_(0, tgt, feat[src]); logit_sum.index_add_(0, tgt, logits[src])
        cnt.index_add_(0, tgt, torch.ones(len(tgt), device=dev))

    with fixed_torch_seed():                 # LitePT's forward pass is stochastic (random serialization order)
        for win in eval_windows(z, radius=radius, max_points=max_points):
            run(win)
        for win in cover_remaining(z, counts, radius, max_points):
            run(win)
    c = cnt.clamp(min=1)[:, None]
    feat = (feat_sum / c).half().cpu().numpy(); logits = (logit_sum / c).half().cpu().numpy()
    seen = (cnt > 0).cpu().numpy()
    assert seen.all(), f"{(~seen).sum()} subset points were not covered by any window"
    a = ctx.assoc
    out = {"zone": z.name, "idx": idx, "labels": z.labels[idx], "w": w, "feat": feat, "logits": logits,
           "img": a.img[idx], "u": a.u[idx], "v": a.v[idx], "desc": a.feat[idx], "image_ids": np.array(a.image_ids),
           "cap": cap, "cap_full": cap_full, "radius": radius, "max_points": max_points}
    if verbose:
        print(f"[protocol] {z.name}: subset {M:,} of {z.n:,} points ({100 * M / z.n:.1f} %), {n_win} windows, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return out


def ensure_subset(ctx, path: str, backbone=None, cfg: Optional[dict] = None, seed: int = 0, cap: int = 500_000,
                  cap_full: int = 2_000_000, verbose: bool = True) -> dict:
    """The stored subset of ctx.zone at `path`, computed (and saved) when missing or built for another image set.
    Shared by the robustness protocol and the feature-bank head trainer: one file per zone, one set of points."""
    sub = None
    if os.path.exists(path):
        sub = load_subset(path)
        if [str(x) for x in sub["image_ids"]] != [str(x) for x in ctx.assoc.image_ids]:
            print(f"[protocol] WARNING: {path} was built for another image set -> recomputing", flush=True)
            sub = None
    if sub is None:
        assert backbone is not None and cfg is not None, f"no subset at {path} and no backbone to build it"
        sub = precompute_subset(backbone, ctx, cfg, np.random.default_rng(seed + 101), cap=cap, cap_full=cap_full, verbose=verbose)
        save_subset(sub, path)
    return sub


def save_subset(sub: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", **sub)
    os.replace(path + ".tmp.npz", path)


def load_subset(path: str) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def subset_rows(sub: dict, device: str) -> Dict[str, torch.Tensor]:
    """Clean association rows of the subset as GPU tensors (same layout as fusion.assoc_rows)."""
    img = torch.from_numpy(sub["img"].astype(np.int64)).to(device)
    return {"img": img.clamp(min=0), "mask": img >= 0,
            "u": torch.from_numpy(sub["u"].astype(np.float32)).to(device),
            "v": torch.from_numpy(sub["v"].astype(np.float32)).to(device),
            "desc": torch.from_numpy(sub["desc"].astype(np.float32)).to(device) * DESC_SCALE.to(device)}


def nearest_views(rows: Dict[str, torch.Tensor], v: int) -> Dict[str, torch.Tensor]:
    """The v nearest valid views of every point. The association keeps up to K views per point in no particular
    order; the first descriptor is the camera-to-point distance. Used to evaluate with fewer views than were stored."""
    V = rows["mask"].shape[1]
    if v >= V:
        return rows
    dist = rows["desc"][..., 0].float().masked_fill(~rows["mask"], float("inf"))
    order = torch.argsort(dist, dim=1, stable=True)[:, :v]
    return {key: torch.gather(t, 1, order if t.dim() == 2 else order[..., None].expand(-1, -1, t.shape[-1]))
            for key, t in rows.items()}


# ----------------------------------------------------------------------------- perturbations
def along_track(S: torch.Tensor, image_ids: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalised along-track coordinate s in [0, 1] of every camera (cumulative path length in capture order =
    sorted image id) and the unit horizontal direction from the first to the last camera."""
    order = np.argsort(np.array(image_ids))
    P = S[order]
    seg = torch.linalg.norm(P[1:] - P[:-1], dim=1)
    cum = torch.cat([torch.zeros(1, device=S.device), torch.cumsum(seg, 0)])
    s = torch.empty(len(image_ids), device=S.device)
    s[torch.as_tensor(order, device=S.device)] = cum / cum[-1].clamp(min=1e-6)
    d = P[-1] - P[0]; d[2] = 0
    n = torch.linalg.norm(d)
    d = d / n if n > 1e-6 else torch.tensor([1.0, 0.0, 0.0], device=S.device)
    return s, d


def drift_cams(S: torch.Tensor, R: torch.Tensor, image_ids: Sequence[str], max_t: float, max_r_deg: float,
               sign: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Systematic drift: position error sign*s*max_t along the track direction and heading error sign*s*max_r
    about the world vertical, s = along-track coordinate of the camera."""
    s, d = along_track(S, image_ids)
    S2 = S + sign * (s * max_t)[:, None] * d[None, :]
    ang = sign * s * (max_r_deg * np.pi / 180.0)
    c, si = torch.cos(ang), torch.sin(ang)
    Rz = torch.zeros(len(s), 3, 3, device=S.device)
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -si, si, c, 1.0
    return S2, R @ Rz.transpose(1, 2)        # camera body rotated by Rz in the world -> camera-from-world = R Rz^T


def shared_cams(S: torch.Tensor, R: torch.Tensor, sigma_t: float, sigma_r_deg: float,
                gen: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """The same pose error for every camera: ONE draw of `perturb_cams` (the i.i.d. model for a single camera at the
    origin with the identity rotation) applied to all of them, so every camera has the marginal distribution of the
    i.i.d. model and the cameras are perfectly correlated. Translation: C_j + dC (the image block is offset against
    the point cloud). Rotation: dR R_j, the same small rotation about the axes of every camera (boresight error)."""
    dS, dR = perturb_cams(torch.zeros(1, 3, device=S.device, dtype=S.dtype),
                          torch.eye(3, device=R.device, dtype=R.dtype)[None], sigma_t, sigma_r_deg, gen)
    return S + dS.to(S.dtype), dR.to(R.dtype) @ R


def perturbed_cams(S: torch.Tensor, R: torch.Tensor, image_ids: Sequence[str], sweep: str, level: float,
                   seed: int, draw: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """One draw of the perturbation `sweep` at `level`; deterministic in `seed` (shared by all models). The drift
    has one degree of freedom, its direction: with `draw` given, even draws take the positive and odd draws the
    negative direction (the protocol evaluates draws 0 and 1); `draw=None` draws the direction from the seed.
    `trans_shared` / `rot_shared`: one error for all cameras (`shared_cams`)."""
    if level == 0.0:
        return S.clone(), R.clone()
    gen = torch.Generator(device=S.device); gen.manual_seed(seed)
    if sweep == "trans":
        return perturb_cams(S, R, sigma_t=level, sigma_r_deg=0.0, gen=gen)
    if sweep == "rot":
        return perturb_cams(S, R, sigma_t=0.0, sigma_r_deg=level, gen=gen)
    if sweep == "trans_shared":
        return shared_cams(S, R, sigma_t=level, sigma_r_deg=0.0, gen=gen)
    if sweep == "rot_shared":
        return shared_cams(S, R, sigma_t=0.0, sigma_r_deg=level, gen=gen)
    if sweep == "drift":
        if draw is None:
            sign = 1.0 if torch.rand(1, generator=gen, device=S.device).item() < 0.5 else -1.0
        else:
            sign = 1.0 if draw % 2 == 0 else -1.0
        return drift_cams(S, R, image_ids, max_t=level, max_r_deg=DRIFT_DEG.get(level, level), sign=sign)
    raise ValueError(f"unknown sweep {sweep!r}")


def draws_for(sweep: str, level: float, draws: int) -> int:
    """Number of evaluations of one level: 1 without noise, both directions for the drift (it has no other
    randomness), `draws` random draws for the i.i.d. sweeps."""
    if level == 0.0:
        return 1
    return min(draws, 2) if sweep == "drift" else draws


_SWEEP_ID = {"trans": 0, "rot": 1, "drift": 2}


def draw_seed(seed: int, sweep: str, level: float, draw: int, zone: Optional[str] = None) -> int:
    """Seed of one perturbation draw: a function of the sweep NAME and the level VALUE, not of their positions in
    the lists, so that runs with other level lists or sweep subsets (e.g. the exact-mode check) perturb the cameras
    identically for the draws they share. A shared-error sweep draws ONE error per zone: its seed also depends on
    `zone`, otherwise every zone of a draw would get the same error (the other sweeps ignore `zone`: their seeds are
    those of every run so far)."""
    sid = _SWEEP_ID.get(sweep, 3 + zlib.crc32(sweep.encode()) % 1_000_003)
    key = [int(seed), sid, int(round(level * 1e6)), int(draw)]
    if zone is not None and sweep in SHARED_SWEEPS:
        key.append(zlib.crc32(str(zone).encode()))
    return int(np.random.SeedSequence(key).generate_state(1)[0])


def poses_from_cams(S: torch.Tensor, R: torch.Tensor, origin: np.ndarray, image_ids: Sequence[str]) -> Dict[str, Pose]:
    Sn = S.detach().cpu().numpy().astype(np.float64) + origin; Rn = R.detach().cpu().numpy().astype(np.float64)
    return {iid: Pose(S=Sn[i], R=Rn[i], opk=np.zeros(3)) for i, iid in enumerate(image_ids)}


# ----------------------------------------------------------------------------- association under a draw
def rows_fast(rows_clean: Dict[str, torch.Tensor], xyz: torch.Tensor, S: torch.Tensor, R: torch.Tensor, cal,
              depths: Optional[torch.Tensor] = None, W: Optional[int] = None, tau: float = 0.5, chunk: int = 1_000_000,
              retest: bool = False) -> Dict[str, torch.Tensor]:
    """Stored views re-projected under (S, R), chunked over points. The view set and the visibility of the clean
    association are kept (a pair is dropped only when the point leaves the image): with wrong poses a pipeline renders
    its depth maps from the same point cloud with the same wrong poses, so what a point can see hardly changes - what
    changes is the pixel it reads. `retest=True` (mode "fast_retest", the first design) also tests the new pixel
    against the CLEAN depth map, which removes exactly the pairs that would read a nearer object by mistake."""
    out = {k: [] for k in ("img", "u", "v", "desc", "mask")}
    for s in range(0, len(xyz), chunk):
        sel = slice(s, min(len(xyz), s + chunk))
        r = reproject_rows({k: v[sel] for k, v in rows_clean.items()}, xyz[sel], S, R, cal,
                           depths if retest else None, W if retest else None, tau=tau)
        for k in out:
            out[k].append(r[k])
    return {k: torch.cat(v) for k, v in out.items()}


def rows_exact(xyz_all: np.ndarray, sub: dict, origin: np.ndarray, S: torch.Tensor, R: torch.Tensor, cal,
               image_ids: Sequence[str], normals_sub: Optional[np.ndarray], k: int = 8, tau: float = 0.5,
               chunk: int = 20_000_000, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Full re-association of the subset under (S, R): depth maps re-rendered from ALL points with the perturbed
    cameras, then projection + visibility + K-nearest view selection for the subset points."""
    d, W, H = depth_maps(xyz_all, S, R, cal, chunk=chunk, verbose=False)
    poses = poses_from_cams(S, R, origin, image_ids)
    a = associate(xyz_all[sub["idx"]], origin, poses, cal, list(image_ids), k=k, tau=tau, normals=normals_sub,
                  chunk=chunk, device=device, verbose=False, depths=d)
    del d
    return assoc_rows(a, np.arange(len(sub["idx"])), device)


def weighted_quantiles(x: torch.Tensor, w: torch.Tensor, qs: Sequence[float]) -> List[float]:
    """Quantiles of x under the weights w (nan when x is empty)."""
    if x.numel() == 0:
        return [float("nan")] * len(qs)
    order = torch.argsort(x)
    xs = x[order]; cw = torch.cumsum(w[order].double(), 0)
    return [float(xs[int(torch.searchsorted(cw, q * cw[-1]).clamp(max=len(xs) - 1))]) for q in qs]


@torch.no_grad()
def pixel_shift(rows_clean: Dict[str, torch.Tensor], xyz: torch.Tensor, S0: torch.Tensor, R0: torch.Tensor,
                S: torch.Tensor, R: torch.Tensor, cal, w: torch.Tensor, labels: torch.Tensor, stride: int = 4) -> Dict[str, float]:
    """How far the stored (point, view) pairs move in the image when the cameras (S0, R0) become (S, R), in
    full-resolution pixels: weighted median and 90th percentile over the pairs that stay in the image, for all
    points and for the points of the electrical classes. Both positions are projected here (the stored pixel
    coordinates are whole pixels and would add ~0.7 px). The weights are the subset weights, so the numbers describe
    the zone and not the class-stratified subset. Every `stride`-th subset point is used. This is the scale on which
    a pose error acts on a model: one cell of the 1/8-resolution feature map is 8 px, one of the probability map 4 px."""
    rc = {k: v[::stride] for k, v in rows_clean.items()}
    pts = xyz[::stride]
    r1 = reproject_rows(rc, pts, S0, R0, cal)
    r2 = reproject_rows(rc, pts, S, R, cal)
    m = r1["mask"] & r2["mask"]
    d = torch.sqrt((r2["u"] - r1["u"]) ** 2 + (r2["v"] - r1["v"]) ** 2)
    ww = w[::stride].float()[:, None].expand_as(d)
    lab = labels[::stride]
    el = torch.zeros_like(lab, dtype=torch.bool)
    for c in ELECTRICAL:
        el |= lab == c
    el = el[:, None].expand_as(d)
    med, p90 = weighted_quantiles(d[m], ww[m], (0.5, 0.9))
    med_el, = weighted_quantiles(d[m & el], ww[m & el], (0.5,))
    out = {"shift_px": med, "shift_px_p90": p90, "shift_px_electrical": med_el}
    return {k: (None if v != v else v) for k, v in out.items()}       # no pair left (e.g. a zone without electrical points): null


# ----------------------------------------------------------------------------- models
class ProtocolModels:
    """The models evaluated side by side on the same rows: 'lidar' (reference), 'imagevote' and trained heads."""

    def __init__(self, heads: Dict[str, nn.Module], device: str = "cuda", imagevote: bool = True, lidar: bool = True):
        self.heads = {n: h.to(device).eval() for n, h in heads.items()}
        self.device = device
        self.names = (["lidar"] if lidar else []) + (["imagevote"] if imagevote else []) + list(self.heads)
        self.max_window = max([h.window for h in self.heads.values() if getattr(h, "uses_feat", True)] + [1])

    @staticmethod
    def from_checkpoints(specs: Dict[str, str], device: str = "cuda") -> Dict[str, nn.Module]:
        """{name: path/to/best.pth} -> {name: head module}; the head type and K come from the checkpoint's cfg."""
        heads = {}
        for name, path in specs.items():
            st = torch.load(path, map_location="cpu", weights_only=False)
            cfg = st["cfg"]
            kw = {"uniform_views": True} if (cfg["head"] == "late" and cfg.get("uniform_views")) else {}
            h = build_head(cfg["head"], k=cfg.get("k", 3), lidar_input=cfg.get("lidar_input", "feat"),
                           image_input=cfg.get("image_input", "feat"), **kw)
            h.load_state_dict(st["head"])
            heads[name] = h
            print(f"[protocol] head {name}: {cfg['head']}" + (f" K={cfg.get('k')}" if cfg["head"] == "window" else "")
                  + f" from {path} (it {st.get('it')}, val {st.get('best', float('nan')):.2f})", flush=True)
        return heads

    @torch.no_grad()
    def predict(self, maps, rows: Dict[str, torch.Tensor], logits: torch.Tensor, feat: torch.Tensor,
                chunk: int = 200_000) -> Dict[str, torch.Tensor]:
        """Per-model predicted class (M,) for the subset given its rows (clean or perturbed). `chunk` is the number
        of points per step for windows up to 3x3; larger windows take proportionally fewer points, so the K x K
        features of a step (chunk x views x K^2 x 32 floats) stay at ~1.8 GB."""
        M = logits.shape[0]
        preds = {n: torch.empty(M, dtype=torch.uint8, device=self.device) for n in self.names}
        if "lidar" in preds:
            preds["lidar"][:] = logits.argmax(1).to(torch.uint8)
        kmax = self.max_window
        chunk = max(10_000, int(chunk * 9 / max(9, kmax * kmax)))
        for s in range(0, M, chunk):
            sel = slice(s, min(M, s + chunk))
            r = {k: v[sel] for k, v in rows.items()}
            img_prob, img_feat = lift(maps, r, window=kmax)
            if "imagevote" in preds:
                p = (img_prob * uniform_weights(r["mask"])[..., None]).sum(1)
                pv = p.argmax(1).to(torch.uint8)
                pv[~r["mask"].any(1)] = IMAGEVOTE_FALLBACK
                preds["imagevote"][sel] = pv
            lg, ft = logits[sel].float(), feat[sel].float()
            for n, h in self.heads.items():
                f_in = window_slice(img_feat, kmax, h.window) if getattr(h, "uses_feat", True) else None
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    out = h(lg, ft, img_prob, f_in, r["desc"], r["mask"])
                preds[n][sel] = out.float().argmax(1).to(torch.uint8)
        return preds


def window_slice(feat_w: Optional[torch.Tensor], k_from: int, k_to: int) -> Optional[torch.Tensor]:
    """(N, V, k_from*k_from, C) window features -> the central k_to×k_to cells ((N, V, C) when k_to == 1)."""
    if feat_w is None or k_from == k_to:
        return feat_w
    if k_from == 1:
        raise ValueError("cannot grow a window")
    N, V, _, C = feat_w.shape
    g = feat_w.reshape(N, V, k_from, k_from, C)
    r0 = (k_from - k_to) // 2
    sub = g[:, :, r0:r0 + k_to, r0:r0 + k_to, :]
    return sub.reshape(N, V, C) if k_to == 1 else sub.reshape(N, V, k_to * k_to, C)


def weighted_confusion(gt: torch.Tensor, pred: torch.Tensor, w: torch.Tensor, n: int = 11) -> np.ndarray:
    """(n, n) float64 confusion matrix with per-point weights (rows = ground truth)."""
    key = gt.long() * n + pred.long()
    cm = torch.bincount(key, weights=w.double(), minlength=n * n)
    return cm.reshape(n, n).cpu().numpy()


# ----------------------------------------------------------------------------- metrics
def iou_from_cm(cm: np.ndarray) -> np.ndarray:
    inter = np.diag(cm); union = cm.sum(0) + cm.sum(1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), np.nan)


def robustness_score(levels: Sequence[float], iou: Sequence[float]) -> float:
    """R = (1/sigma_max) * integral_0^sigma_max IoU(sigma)/IoU(0) d sigma, trapezoid over the levels (nan-safe)."""
    x = np.asarray(levels, float); y = np.asarray(iou, float)
    if len(x) < 2 or not np.isfinite(y[0]) or y[0] <= 0:
        return float("nan")
    y = y / y[0]
    m = np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(y[m], x[m]) / (x[m][-1] - x[m][0]))


def summarize(results: dict) -> dict:
    """Pool the per-zone confusion matrices, average IoU over draws, score every (model, sweep, class)."""
    meta = results["meta"]; runs = results["runs"]
    pooled: Dict[Tuple[str, str, float, int], np.ndarray] = {}
    per_zone: Dict[Tuple[str, str, float, int, str], np.ndarray] = {}
    for r in runs:
        key = (r["model"], r["sweep"], float(r["level"]), int(r["draw"]))
        cm = np.asarray(r["cm"], float)
        pooled[key] = pooled.get(key, 0) + cm
        per_zone[key + (r["zone"],)] = cm
    curves, scores = [], []
    for model in meta["models"]:
        for sweep, levels in meta["sweeps"].items():
            per_level = []
            for lv in levels:
                draws = sorted({k[3] for k in pooled if k[0] == model and k[1] == sweep and k[2] == float(lv)})
                ious = np.array([iou_from_cm(pooled[(model, sweep, float(lv), d)]) for d in draws])   # (D, 11)
                if len(ious) == 0:
                    per_level.append(np.full(11, np.nan)); continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)          # classes absent from every zone -> nan
                    mean, std = np.nanmean(ious, 0), (np.nanstd(ious, 0) if len(ious) > 1 else np.zeros(11))
                per_level.append(mean)
                for c, name in enumerate(CLASS_NAMES):
                    curves.append({"model": model, "sweep": sweep, "level": float(lv), "class": name,
                                   "iou_mean": float(mean[c]), "iou_std": float(std[c]), "draws": len(ious)})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    m_all = float(np.nanmean(mean)); m_el = float(np.nanmean(mean[list(ELECTRICAL)]))
                curves.append({"model": model, "sweep": sweep, "level": float(lv), "class": "mean_all", "iou_mean": m_all,
                               "iou_std": float(np.nanstd(np.nanmean(ious, 1))) if len(ious) > 1 else 0.0, "draws": len(ious)})
                curves.append({"model": model, "sweep": sweep, "level": float(lv), "class": "mean_electrical", "iou_mean": m_el,
                               "iou_std": float(np.nanstd(np.nanmean(ious[:, list(ELECTRICAL)], 1))) if len(ious) > 1 else 0.0,
                               "draws": len(ious)})
            per_level = np.array(per_level)                                                    # (L, 11)
            ref = REFERENCE_LEVEL.get(sweep, levels[-1])
            ref_i = int(np.argmin(np.abs(np.array(levels) - ref)))
            groups = {name: per_level[:, c] for c, name in enumerate(CLASS_NAMES)}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                groups["mean_all"] = np.nanmean(per_level, 1); groups["mean_electrical"] = np.nanmean(per_level[:, list(ELECTRICAL)], 1)
            for name, y in groups.items():
                scores.append({"model": model, "sweep": sweep, "class": name, "clean_iou": float(y[0]),
                               "R": robustness_score(levels, y), "ref_level": float(levels[ref_i]),
                               "drop_ref": float(y[0] - y[ref_i]) if np.isfinite(y[0]) and np.isfinite(y[ref_i]) else float("nan"),
                               "iou_max_level": float(y[-1])})
    # how far a projected point moves at every level: mean over zones and draws of the per-draw statistics
    displacement = []
    by_level: Dict[Tuple[str, float], List[dict]] = {}
    for d in results.get("draws", []):
        by_level.setdefault((d["sweep"], float(d["level"])), []).append(d)
    for sweep, levels in meta["sweeps"].items():
        for lv in levels:
            rows = by_level.get((sweep, float(lv)), [])
            if not rows:
                continue
            row = {"sweep": sweep, "level": float(lv), "n": len(rows)}
            for key in ("shift_px", "shift_px_p90", "shift_px_electrical", "coverage"):
                v = np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)
                row[key] = float(np.nanmean(v)) if np.isfinite(v).any() else None
            displacement.append(row)
    # one confusion matrix per (model, sweep, level): pooled over zones, mean over draws (rows = ground truth)
    confusion = []
    by_model_level: Dict[Tuple[str, str, float], List[np.ndarray]] = {}
    for (model, sweep, lv, _), cm in pooled.items():
        by_model_level.setdefault((model, sweep, lv), []).append(cm)
    for model in meta["models"]:
        for sweep, levels in meta["sweeps"].items():
            for lv in levels:
                cms = by_model_level.get((model, sweep, float(lv)))
                if cms:
                    confusion.append({"model": model, "sweep": sweep, "level": float(lv), "draws": len(cms),
                                      "cm": np.round(np.mean(cms, 0), 1).tolist()})
    return {"meta": meta, "curves": curves, "scores": scores, "displacement": displacement, "confusion": confusion}


def compare_runs(a: dict, b: dict) -> List[dict]:
    """Two protocol runs (results dicts, e.g. fast and exact mode) on the evaluations they share - same model, sweep,
    level, draw and zone, i.e. the same perturbed cameras. Per (model, sweep, level): confusions pooled over the
    shared (zone, draw) pairs -> mean IoU over all / the electrical classes for both runs, the class with the largest
    IoU difference, and the mean coverage (share of points with at least one valid view)."""
    def index(res):
        return {(r["model"], r["sweep"], float(r["level"]), int(r["draw"]), r["zone"]): r for r in res["runs"]}
    ia, ib = index(a), index(b)
    groups: Dict[Tuple[str, str, float], list] = {}
    for key in sorted(set(ia) & set(ib)):
        groups.setdefault(key[:3], []).append(key)
    rows = []
    for (model, sweep, level), keys in groups.items():
        out = {"model": model, "sweep": sweep, "level": level, "n": len(keys),
               "zones": sorted({k[4] for k in keys}), "draws": sorted({k[3] for k in keys})}
        ious = {}
        for tag, idx in (("a", ia), ("b", ib)):
            cm = sum(np.asarray(idx[k]["cm"], float) for k in keys)
            iou = iou_from_cm(cm); ious[tag] = iou
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[f"mean_all_{tag}"] = float(np.nanmean(iou)); out[f"mean_electrical_{tag}"] = float(np.nanmean(iou[list(ELECTRICAL)]))
            out[f"coverage_{tag}"] = float(np.mean([idx[k].get("coverage", np.nan) for k in keys]))
            out[f"iou_{tag}"] = [None if np.isnan(v) else float(v) for v in iou]
        d = ious["b"] - ious["a"]
        if np.isfinite(d).any():
            c = int(np.nanargmax(np.abs(d)))
            out["max_diff_class"], out["max_diff"] = CLASS_NAMES[c], float(d[c])
        else:
            out["max_diff_class"], out["max_diff"] = "", float("nan")
        rows.append(out)
    return rows


def write_summary(summary: dict, out_dir: str) -> None:
    import csv
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({k: v for k, v in summary.items() if k != "confusion"}, fh, indent=1)
    if summary.get("confusion"):                     # compact: ~0.4 MB for 24 models, readable without results.json
        with open(os.path.join(out_dir, "confusion.json"), "w") as fh:
            json.dump({"class_names": summary["meta"]["class_names"], "rows": summary["confusion"]}, fh, separators=(",", ":"))
    for name in ("curves", "scores"):
        rows = summary[name]
        if not rows:
            continue
        with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------------------- driver
def miou_present(cm) -> float:
    """Mean IoU [%] over the classes with ground-truth points in `cm`. A class that a zone does not contain but a
    model predicts a few points of has an IoU of 0 and would move a plain mean by 1/11 of the scale."""
    cm = np.asarray(cm, float)
    present = cm.sum(1) > 0
    return float(100 * np.nanmean(iou_from_cm(cm)[present])) if present.any() else float("nan")


def meta_conflict(stored: dict, new: dict) -> Optional[str]:
    """Why a stored results.json cannot be continued with the current settings (None: it can). Zones may be added;
    models, sweeps, draws, mode, tau, seed, the number of views and the protocol version must be the same, otherwise
    the pooled confusion matrices of a (model, level, draw) would cover different zones."""
    diff = []
    for k in ("version", "models", "sweeps", "draws", "mode", "tau", "seed", "views"):
        a = stored.get(k, {"version": 1, "views": 8}.get(k))            # files written before a key existed
        a, b = json.loads(json.dumps(a)), json.loads(json.dumps(new.get(k, {"views": 8}.get(k))))
        if a != b:
            diff.append(f"{k}: stored {a} / now {b}")
    return "; ".join(diff) or None


def run_protocol(models: ProtocolModels, zone_loader, zones: Sequence[str], out_dir: str, sweeps: Dict[str, List[float]],
                 draws: int = 10, mode: str = "fast", tau: float = 0.5, seed: int = 0, k: int = 8,
                 subset_dir: Optional[str] = None, backbone=None, cfg: Optional[dict] = None, cap: int = 500_000,
                 cap_full: int = 2_000_000, verbose: bool = True, on_draw=None) -> dict:
    """zone_loader(zone) -> (ctx, normals (N,3) or None); ctx needs its depth maps only for mode "fast_retest".
    `k` = views per point: the k nearest of the stored views (clean level, fast modes) and the k nearest visible
    ones of the re-association (exact mode); the heads were trained with 8.
    Resume-safe per zone: results.json is re-read and finished zones are skipped (the settings must be the ones
    the file was started with, see `meta_conflict`). Returns the results dict.
    `on_draw(zone=, sweep=, level=, draw=, preds=, xyz=, labels=, weights=)` is called once per evaluation with the
    predictions of every model (name -> class per subset point) and the subset itself; it must not change them. The
    results do not depend on it (gridnet.qualitative.CropDump saves crops for figures through it). A zone that is
    skipped because results.json already holds it is not evaluated again, so the hook is not called for it."""
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.json")
    meta = {"version": PROTOCOL_VERSION, "models": list(models.names), "sweeps": {k: [float(x) for x in v] for k, v in sweeps.items()},
            "draws": int(draws), "drift_draws": min(int(draws), 2), "mode": mode, "tau": tau, "seed": seed, "views": int(k), "zones": list(zones),
            "class_names": CLASS_NAMES, "reference": REFERENCE_LEVEL, "drift_deg": DRIFT_DEG}
    if os.path.exists(res_path):
        with open(res_path) as fh:
            results = json.load(fh)
        why = meta_conflict(results["meta"], meta)
        if why:
            raise ValueError(f"{res_path} was written with other settings ({why}). A run is continued only with the same "
                             f"models, sweeps, draws, mode and protocol version: use another output folder (TAG).")
        results["meta"]["zones"] = list(dict.fromkeys(list(results["meta"].get("zones", [])) + list(zones)))
    else:
        results = {"meta": meta, "runs": []}
    results.setdefault("draws", [])                  # per (zone, sweep, level, draw): coverage + pixel displacement
    done = {r["zone"] for r in results["runs"]}
    sweep_names = list(sweeps)
    for zone in zones:
        if zone in done:
            if verbose:
                print(f"[protocol] {zone}: already in {res_path} - skip", flush=True)
            continue
        t0 = time.time()
        ctx, normals = zone_loader(zone)
        dev = models.device
        sub_path = os.path.join(subset_dir or out_dir, zone, "subset.npz")
        sub = ensure_subset(ctx, sub_path, backbone, cfg, seed=seed, cap=cap, cap_full=cap_full, verbose=verbose)
        gt = torch.from_numpy(sub["labels"].astype(np.int64)).to(dev)
        w = torch.from_numpy(sub["w"]).to(dev)
        logits = torch.from_numpy(sub["logits"]).to(dev); feat = torch.from_numpy(sub["feat"]).to(dev)
        rows_clean = nearest_views(subset_rows(sub, dev), k)       # fewer views than stored: the k nearest (views ablation)
        xyz_sub = torch.from_numpy(ctx.zone.xyz[sub["idx"]]).to(dev)
        normals_sub = None if normals is None else normals[sub["idx"]]
        zone_runs, zone_draws = [], []
        for sweep in sweep_names:
            for level in sweeps[sweep]:
                for d in range(draws_for(sweep, level, draws)):
                    t1 = time.time()
                    S2, R2 = perturbed_cams(ctx.S, ctx.R, ctx.assoc.image_ids, sweep, level,
                                            draw_seed(seed, sweep, level, d, zone=zone), draw=d)
                    if level == 0.0:
                        rows = rows_clean
                    elif mode in ("fast", "fast_retest"):
                        rows = rows_fast(rows_clean, xyz_sub, S2, R2, ctx.cal, ctx.depths, ctx.W, tau=tau,
                                         retest=mode == "fast_retest")
                    elif mode == "exact":
                        rows = rows_exact(ctx.zone.xyz, sub, ctx.origin, S2, R2, ctx.cal, ctx.assoc.image_ids, normals_sub,
                                          k=k, tau=tau, device=dev)
                    else:
                        raise ValueError(mode)
                    preds = models.predict(ctx.maps, rows, logits, feat)
                    if on_draw is not None:
                        on_draw(zone=zone, sweep=sweep, level=float(level), draw=d, preds=preds, xyz=xyz_sub, labels=gt, weights=w)
                    cov = float(rows["mask"].any(1).float().mean())
                    shift = ({"shift_px": 0.0, "shift_px_p90": 0.0, "shift_px_electrical": 0.0} if level == 0.0
                             else pixel_shift(rows_clean, xyz_sub, ctx.S, ctx.R, S2, R2, ctx.cal, w, gt))
                    zone_draws.append(dict({"zone": zone, "sweep": sweep, "level": float(level), "draw": d, "coverage": cov}, **shift))
                    for name, p in preds.items():
                        zone_runs.append({"model": name, "sweep": sweep, "level": float(level), "draw": d, "zone": zone,
                                          "coverage": cov, "cm": np.round(weighted_confusion(gt, p, w), 1).tolist()})
                    if verbose:                                      # zone mIoU over the classes the zone contains
                        ious = {name: miou_present(zone_runs[-len(preds) + i]["cm"]) for i, name in enumerate(preds)}
                        px = float("nan") if shift["shift_px"] is None else shift["shift_px"]
                        print(f"[protocol] {zone} {sweep} {level:g} draw {d}: coverage {100 * cov:.1f} %  shift {px:.1f} px  "
                              + "  ".join(f"{n} {v:.1f}" for n, v in ious.items()) + f"  ({time.time() - t1:.1f} s)", flush=True)
        results["runs"] += zone_runs
        results["draws"] += zone_draws
        with open(res_path + ".tmp", "w") as fh:
            json.dump(results, fh)
        os.replace(res_path + ".tmp", res_path)
        if verbose:
            print(f"[protocol] {zone}: {len(zone_runs)} evaluations in {(time.time() - t0) / 60:.1f} min -> {res_path}", flush=True)
        ctx.free(); del logits, feat, rows_clean, xyz_sub
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(results)
    write_summary(summary, out_dir)
    with gzip.open(res_path + ".gz.tmp", "wt") as fh:           # results.json grows by ~0.6 MB per model: the copy stays small
        json.dump(results, fh, separators=(",", ":"))
    os.replace(res_path + ".gz.tmp", res_path + ".gz")
    return results
