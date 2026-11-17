"""ImageVote: lift the cached 2D class probabilities to the points through the association table and
average them over the selected views (uniform weights). Also the generic "read the 2D cache at the
associated pixels" primitive that the fusion heads reuse.
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch

from ..preprocess.association import Association
from .maps2d import load_prob_fp16


def load_prob(cache2d_dir: str, zone: str, image_id: str) -> np.ndarray:
    return load_prob_fp16(cache2d_dir, zone, image_id)    # (11, H/4, W/4) float16, fp16 or compact cache


@torch.no_grad()
def vote(assoc: Association, cache2d_dir: str, zone: str, ds: int = 4, device: str = "cuda",
         weights: Optional[np.ndarray] = None, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (prob (N, 11) float32 = per-point mean of the view probabilities, n_votes (N,) int32).
    `weights` (N, K) optional per-view weights (default uniform)."""
    n, k = assoc.img.shape
    img_t = torch.as_tensor(assoc.img.astype(np.int16), device=device)
    u_t = torch.as_tensor(assoc.u.astype(np.int32), device=device)
    v_t = torch.as_tensor(assoc.v.astype(np.int32), device=device)
    w_t = None if weights is None else torch.as_tensor(np.asarray(weights, np.float32), device=device)
    votes = torch.zeros((n, 11), dtype=torch.float32, device=device)
    wsum = torch.zeros(n, dtype=torch.float32, device=device)
    counts = torch.zeros(n, dtype=torch.int32, device=device)
    for j, iid in enumerate(assoc.image_ids):
        hit = torch.nonzero(img_t == j, as_tuple=True)
        if len(hit[0]) == 0:
            continue
        rows, slots = hit
        prob = torch.as_tensor(load_prob(cache2d_dir, zone, iid), device=device)   # (11, H4, W4) fp16
        H4, W4 = prob.shape[1:]
        cu = (u_t[rows, slots] // ds).clamp(0, W4 - 1)
        cv = (v_t[rows, slots] // ds).clamp(0, H4 - 1)
        p = prob[:, cv, cu].float().T                                              # (M, 11)
        w = torch.ones(len(rows), device=device) if w_t is None else w_t[rows, slots]
        votes.index_add_(0, rows, p * w[:, None])
        wsum.index_add_(0, rows, w)
        counts.index_add_(0, rows, torch.ones(len(rows), dtype=torch.int32, device=device))
        if verbose and (j + 1) % 25 == 0:
            print(f"\r[vote] {j + 1}/{len(assoc.image_ids)} images", end="", flush=True)
    if verbose:
        print()
    prob = votes / wsum.clamp(min=1e-6)[:, None]
    return prob.cpu().numpy(), counts.cpu().numpy()
