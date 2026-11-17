"""Fusion heads on frozen backbones (paper 1, stage 3).

Everything a head sees comes through two gathers from the per-zone 2D cache resident on the GPU
(`ZoneMaps`): the centre cell of each associated view (class probabilities @1/4 res, features @1/8 res)
and, for the window head, the K×K feature cells around it. Views are weighted by a small MLP on the
per-view descriptors (distance, incidence, depth residual, radial position, depth).

Heads (all end in 11-class logits per point):
  LateFusionHead   softmax(LiDAR logits) ⊕ view-weighted image probabilities        -> MLP
  ConcatHead       LiDAR feature (72) ⊕ view-weighted image feature (32) ⊕ probs    -> MLP
  WindowAttnHead   as ConcatHead, but the image feature of each view is attention-pooled over the
                   K×K cells around the projected location (query = LiDAR feature)
"""
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .maps2d import PROB_SCALE, load_maps

DESC_SCALE = torch.tensor([1 / 100.0, 1.0, 1 / 0.5, 1.0, 1 / 100.0])   # dist m, cos, resid m, radial, depth m


# ----------------------------------------------------------------------------- 2D maps on the GPU
class ZoneMaps:
    """The 2D cache of one zone as GPU tensors: prob uint8 (n_img, 11, H4, W4), feat int8 (n_img, 32, H8, W8),
    feat_scale float32 (n_img, 32). `image_ids` fixes the image index used by the association."""

    def __init__(self, cache2d_dir: str, zone: str, image_ids: Sequence[str], device: str = "cuda",
                 verbose: bool = True, with_feat: bool = True):
        self.image_ids = list(image_ids)
        self.device = device
        probs, feats, scales = [], [], []
        for i, iid in enumerate(self.image_ids):
            m = load_maps(cache2d_dir, zone, iid, what=("prob", "feat") if with_feat else ("prob",))
            probs.append(torch.from_numpy(m["prob"]))
            if with_feat:
                feats.append(torch.from_numpy(m["feat"])); scales.append(torch.from_numpy(m["feat_scale"]))
            if verbose and (i + 1) % 50 == 0:
                print(f"\r[maps] {zone}: {i + 1}/{len(self.image_ids)} images", end="", flush=True)
        if verbose:
            print(f"\r[maps] {zone}: {len(self.image_ids)} images loaded", flush=True)
        self.prob = torch.stack(probs).to(device)                                   # uint8
        self.H4, self.W4 = self.prob.shape[-2:]
        if with_feat:
            self.feat = torch.stack(feats).to(device)                               # int8
            self.scale = torch.stack(scales).to(device)                             # float32 (n_img, 32)
            self.H8, self.W8 = self.feat.shape[-2:]
        else:
            self.feat = None

    @property
    def n_images(self) -> int:
        return int(self.prob.shape[0])

    def gather_prob(self, img: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """(M,) image index + full-res pixel coords -> (M, 11) float probabilities at the 1/4-res cell."""
        cu = (u // 4).long().clamp_(0, self.W4 - 1); cv = (v // 4).long().clamp_(0, self.H4 - 1)
        return self.prob[img, :, cv, cu].float() / PROB_SCALE

    def gather_feat(self, img: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """(M, 32) float features at the 1/8-res cell."""
        cu = (u // 8).long().clamp_(0, self.W8 - 1); cv = (v // 8).long().clamp_(0, self.H8 - 1)
        return self.feat[img, :, cv, cu].float() * self.scale[img]

    def gather_window(self, img: torch.Tensor, u: torch.Tensor, v: torch.Tensor, k: int) -> torch.Tensor:
        """(M, k*k, 32) float features of the k×k cells (1/8 res) around the projected location; cells past
        the image border are clamped to the border (so they repeat the edge cell)."""
        r = k // 2
        cu = (u // 8).long(); cv = (v // 8).long()
        offs = torch.arange(-r, r + 1, device=img.device)
        gy = (cv[:, None, None] + offs[None, :, None]).clamp_(0, self.H8 - 1)      # (M, k, 1)
        gx = (cu[:, None, None] + offs[None, None, :]).clamp_(0, self.W8 - 1)      # (M, 1, k)
        gy = gy.expand(-1, k, k).reshape(len(img), -1); gx = gx.expand(-1, k, k).reshape(len(img), -1)
        win = self.feat[img[:, None], :, gy, gx]                                     # (M, k*k, 32) int8
        return win.float() * self.scale[img][:, None, :]


# ----------------------------------------------------------------------------- association rows -> GPU
def assoc_rows(assoc, idx: np.ndarray, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Slice the per-point association table for the points `idx` and move it to the GPU.
    Returns img (N, V) long (-1 = empty), u/v (N, V) float32 (full-res pixels), desc (N, V, 5) float32
    (scaled), mask (N, V) bool."""
    img = torch.from_numpy(assoc.img[idx].astype(np.int64)).to(device)
    mask = img >= 0
    u = torch.from_numpy(assoc.u[idx].astype(np.float32)).to(device)
    v = torch.from_numpy(assoc.v[idx].astype(np.float32)).to(device)
    desc = torch.from_numpy(assoc.feat[idx].astype(np.float32)).to(device) * DESC_SCALE.to(device)
    return {"img": img.clamp(min=0), "u": u, "v": v, "desc": desc, "mask": mask}


def lift(maps: ZoneMaps, rows: Dict[str, torch.Tensor], window: int = 1, chunk: int = 200_000):
    """Read the 2D cache at every (point, view) pair. Returns prob (N, V, 11), feat (N, V, 32) or, with
    window > 1, feat (N, V, k*k, 32); empty slots are zero. Chunked over points."""
    img, u, v, mask = rows["img"], rows["u"], rows["v"], rows["mask"]
    N, V = img.shape
    probs, feats = [], []
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        i = img[s:e].reshape(-1); uu = u[s:e].reshape(-1); vv = v[s:e].reshape(-1); m = mask[s:e].reshape(-1)
        p = maps.gather_prob(i, uu, vv) * m[:, None]
        probs.append(p.reshape(e - s, V, -1))
        if maps.feat is not None:
            if window > 1:
                f = maps.gather_window(i, uu, vv, window) * m[:, None, None]
                feats.append(f.reshape(e - s, V, window * window, -1))
            else:
                f = maps.gather_feat(i, uu, vv) * m[:, None]
                feats.append(f.reshape(e - s, V, -1))
    return torch.cat(probs), (torch.cat(feats) if feats else None)


# ----------------------------------------------------------------------------- heads
def mlp(cin: int, hidden: int, cout: int, depth: int = 2, dropout: float = 0.1) -> nn.Sequential:
    layers, c = [], cin
    for _ in range(depth):
        layers += [nn.Linear(c, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]; c = hidden
    layers.append(nn.Linear(c, cout))
    return nn.Sequential(*layers)


class ViewWeighter(nn.Module):
    """Softmax weights over the V views of a point from the per-view descriptors (empty slots get 0)."""

    def __init__(self, n_desc: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_desc, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, desc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.net(desc).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(s, dim=1) * mask
        return w                                                  # (N, V); all-zero rows for points without a view


def weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x (N, V, C), w (N, V) -> (N, C)."""
    return (x * w.unsqueeze(-1)).sum(1)


class LateFusionHead(nn.Module):
    """Decision-level fusion: LiDAR class probabilities ⊕ view-weighted image probabilities -> MLP."""
    uses_feat = False
    window = 1

    def __init__(self, num_classes: int = 11, hidden: int = 64, uniform_views: bool = False):
        super().__init__()
        self.weighter = None if uniform_views else ViewWeighter()
        self.net = mlp(2 * num_classes + 2, hidden, num_classes)

    def forward(self, lidar_logits, lidar_feat, img_prob, img_feat, desc, mask):
        w = uniform_weights(mask) if self.weighter is None else self.weighter(desc, mask)
        p_img = weighted_mean(img_prob, w)
        has = mask.any(1, keepdim=True).float(); nv = mask.float().sum(1, keepdim=True) / mask.shape[1]
        return self.net(torch.cat([torch.softmax(lidar_logits.float(), 1), p_img, has, nv], 1))


class ConcatHead(nn.Module):
    """Feature-level fusion: LiDAR representation ⊕ view-weighted image feature ⊕ image probabilities -> MLP.
    Two switches select what the head sees of each branch, for the input ablation of the paper:
    `lidar_input`: "feat" = the 72-d decoder feature of the LiDAR branch (classic feature concatenation), "prob" =
    only its class probabilities, as in the late head; `image_input`: "feat" = image feature + image
    probabilities, "prob" = image probabilities only. (concat_v1 and window3_v1, both "feat"/"feat", lose the water
    class on the validation zones that the late head keeps; these switches isolate which input is responsible.)"""
    uses_feat = True
    window = 1

    def __init__(self, lidar_dim: int = 72, img_dim: int = 32, num_classes: int = 11, hidden: int = 128,
                 lidar_input: str = "feat", image_input: str = "feat"):
        super().__init__()
        assert lidar_input in ("feat", "prob"), lidar_input
        assert image_input in ("feat", "prob"), image_input
        self.lidar_input, self.image_input = lidar_input, image_input
        if lidar_input == "prob":
            lidar_dim = num_classes
        self.lidar_dim = lidar_dim
        self.uses_feat = image_input == "feat"            # False: the 2D feature maps are not needed at all
        self.weighter = ViewWeighter()
        self.norm_l = nn.LayerNorm(lidar_dim) if lidar_input == "feat" else nn.Identity()
        self.norm_i = nn.LayerNorm(img_dim) if self.uses_feat else None
        self.net = mlp(lidar_dim + (img_dim if self.uses_feat else 0) + num_classes + 1, hidden, num_classes)

    def lidar_repr(self, lidar_logits, lidar_feat):
        if self.lidar_input == "prob":
            return torch.softmax(lidar_logits.float(), 1)
        return self.norm_l(lidar_feat.float())

    def image_feature(self, lidar_repr, img_feat, desc, mask, w):
        return weighted_mean(img_feat, w)

    def forward(self, lidar_logits, lidar_feat, img_prob, img_feat, desc, mask):
        w = self.weighter(desc, mask)
        lr = self.lidar_repr(lidar_logits, lidar_feat)
        p_img = weighted_mean(img_prob, w)
        has = mask.any(1, keepdim=True).float()
        parts = [lr]
        if self.uses_feat:
            parts.append(self.norm_i(self.image_feature(lr, img_feat, desc, mask, w)) * has)
        return self.net(torch.cat(parts + [p_img, has], 1))


class WindowAttnHead(ConcatHead):
    """K×K window attention: for every (point, view) the image feature is an attention-weighted sum over
    the K×K cells around the projected location, with the LiDAR representation as query and a learned offset
    embedding on keys and values, so a point can pick the cell that matches its own geometry instead of
    trusting the cell that the (possibly wrong) pose indicates. K = 1 reduces to ConcatHead."""
    uses_feat = True

    def __init__(self, k: int = 3, lidar_dim: int = 72, img_dim: int = 32, num_classes: int = 11,
                 hidden: int = 128, d_attn: int = 32, lidar_input: str = "feat"):
        super().__init__(lidar_dim, img_dim, num_classes, hidden, lidar_input)
        self.window = k
        self.q = nn.Linear(self.lidar_dim, d_attn)
        self.k = nn.Linear(img_dim, d_attn)
        self.pos_k = nn.Parameter(torch.zeros(k * k, d_attn))
        self.pos_v = nn.Parameter(torch.zeros(k * k, img_dim))
        nn.init.normal_(self.pos_k, std=0.02); nn.init.normal_(self.pos_v, std=0.02)
        self.scale = d_attn ** -0.5

    def image_feature(self, lidar_repr, img_feat, desc, mask, w):
        if img_feat.dim() == 3:                                                # K = 1: a single cell per view
            img_feat = img_feat[:, :, None, :]
        N, V, KK, C = img_feat.shape                                           # (N, V, k*k, C)
        q = self.q(lidar_repr)                                                # (N, d)
        keys = self.k(img_feat) + self.pos_k                                  # (N, V, KK, d)
        att = torch.einsum("nd,nvkd->nvk", q, keys) * self.scale
        att = torch.softmax(att, dim=-1)                                      # (N, V, KK)
        vals = img_feat + self.pos_v                                          # (N, V, KK, C)
        pooled = torch.einsum("nvk,nvkc->nvc", att, vals)                     # (N, V, C)
        self.last_attention = att.detach()
        return weighted_mean(pooled, w)


def drop_lidar(lidar_logits: torch.Tensor, lidar_feat: Optional[torch.Tensor], p: float,
               gen: Optional[torch.Generator] = None):
    """Modality dropout on the LiDAR branch: for a random fraction p of the points the head gets no LiDAR
    evidence (zero logits = uniform class probabilities, zero feature), so it has to learn every class from the
    image evidence as well. p = 1 is the "image-only" ablation at test time. Returns (logits, feat, dropped)."""
    if p <= 0:
        return lidar_logits, lidar_feat, None
    m = torch.rand(lidar_logits.shape[0], device=lidar_logits.device, generator=gen) < p
    lf = None if lidar_feat is None else lidar_feat.masked_fill(m[:, None], 0.0)
    return lidar_logits.masked_fill(m[:, None], 0.0), lf, m


def uniform_weights(mask: torch.Tensor) -> torch.Tensor:
    n = mask.float().sum(1, keepdim=True).clamp(min=1)
    return mask.float() / n


def build_head(name: str, k: int = 3, **kw) -> nn.Module:
    """late | concat | window; `lidar_input` ("feat" | "prob") applies to concat and window, `image_input`
    ("feat" | "prob") to concat only (a window head without image features would have nothing to attend over)."""
    if name == "late":
        kw.pop("lidar_input", None); kw.pop("image_input", None)
        return LateFusionHead(**kw)
    if name == "concat":
        return ConcatHead(**kw)
    if name == "window":
        if kw.pop("image_input", "feat") != "feat":
            raise ValueError("the window head needs image features: image_input='prob' is a concat-head option")
        return WindowAttnHead(k=k, **kw)
    raise ValueError(f"unknown head {name!r} (late | concat | window)")


# ----------------------------------------------------------------------------- pose-noise re-projection
def perturb_cams(S: torch.Tensor, R: torch.Tensor, sigma_t: float, sigma_r_deg: float,
                 gen: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """i.i.d. pose noise on every camera: translation ~ N(0, sigma_t^2 I) (m), rotation by a random axis-angle
    with per-axis angles ~ N(0, sigma_r^2) (deg). S (n, 3), R (n, 3, 3) -> perturbed copies."""
    n, dev = S.shape[0], S.device
    dS = torch.randn(n, 3, device=dev, generator=gen) * sigma_t
    w = torch.randn(n, 3, device=dev, generator=gen) * (sigma_r_deg * np.pi / 180.0)
    th = torch.linalg.norm(w, dim=1).clamp(min=1e-12)
    a = w / th[:, None]
    K = torch.zeros(n, 3, 3, device=dev)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -a[:, 2], a[:, 1], a[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -a[:, 0], -a[:, 1], a[:, 0]
    I = torch.eye(3, device=dev).expand(n, 3, 3)
    dR = I + torch.sin(th)[:, None, None] * K + (1 - torch.cos(th))[:, None, None] * (K @ K)
    return S + dS, dR @ R


def project_pairs_t(xyz: torch.Tensor, S: torch.Tensor, R: torch.Tensor, cal, conv=(1, -1, 1), fov_margin=1.15):
    """Per-pair projection: xyz (M, 3), S (M, 3), R (M, 3, 3) -> u, v, z, ok (same model as assoc_gpu.project_t)."""
    sx, sy, dz = conv
    P = torch.einsum("mij,mj->mi", R, xyz - S)
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


def reproject_rows(rows: Dict[str, torch.Tensor], xyz: torch.Tensor, S: torch.Tensor, R: torch.Tensor, cal,
                   depths: Optional[torch.Tensor] = None, W: Optional[int] = None, ds: int = 4, tau: float = 0.5):
    """Recompute (u, v) of every (point, view) pair under the camera set (S, R) -- the view set is kept, pairs
    that leave the frame or fail the visibility test against the (clean) depth maps are masked out.
    Used for pose-noise augmentation and for the fast protocol approximation. Returns new rows."""
    img, mask = rows["img"], rows["mask"]
    N, V = img.shape
    flat = img.reshape(-1); m = mask.reshape(-1)
    pts = xyz[:, None, :].expand(N, V, 3).reshape(-1, 3)
    u, v, z, ok = project_pairs_t(pts, S[flat], R[flat], cal)
    ok = ok & m
    if depths is not None:
        lin = (v / ds).long().clamp(min=0) * W + (u / ds).long().clamp(min=0)
        lin = torch.where(ok, lin, torch.zeros_like(lin))
        d = depths[flat, lin].float()
        ok = ok & ((z - d) <= tau)
    out = dict(rows)
    out["u"] = torch.where(ok, u, rows["u"].reshape(-1)).reshape(N, V)
    out["v"] = torch.where(ok, v, rows["v"].reshape(-1)).reshape(N, V)
    out["mask"] = ok.reshape(N, V)
    return out
