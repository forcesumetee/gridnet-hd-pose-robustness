"""Point-cloud geometry: voxel grid subsampling, local PCA features, height above ground.

Everything is numpy/scipy so it runs on a laptop; heavy steps are chunked.
"""
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree


def voxel_grid(xyz: np.ndarray, size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Grid subsampling. Returns (keep, inverse): `keep` indexes one representative point per
    occupied voxel (the point closest to the voxel centre); `inverse[i]` gives, for every original
    point i, the index into `keep` of its voxel — use it to lift per-voxel predictions back."""
    origin = xyz.min(axis=0)
    q = np.floor((xyz - origin) / size).astype(np.int64)
    # pack to a single key (zone extents << 2^21 voxels per axis at 2 cm)
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    uniq, inverse = np.unique(key, return_inverse=True)
    centre = (q + 0.5) * size + origin
    d2 = ((xyz - centre) ** 2).sum(axis=1)
    # representative = argmin distance to centre within each voxel
    order = np.lexsort((d2, inverse))            # sort by voxel, then by distance
    first = np.ones(len(order), dtype=bool)
    first[1:] = inverse[order][1:] != inverse[order][:-1]
    keep = order[first]
    return keep, inverse


def majority_labels(labels: np.ndarray, inverse: np.ndarray, n_vox: int, ignore: int = 255) -> np.ndarray:
    """Per-voxel majority label (ignore label loses to any real label)."""
    out = np.full(n_vox, ignore, dtype=np.uint8)
    valid = labels != ignore
    if not valid.any():
        return out
    pair = inverse[valid].astype(np.int64) * 256 + labels[valid]
    u, cnt = np.unique(pair, return_counts=True)
    vox = u // 256; lab = (u % 256).astype(np.uint8)
    order = np.lexsort((-cnt, vox))               # per voxel, highest count first
    vox_o, lab_o = vox[order], lab[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = vox_o[1:] != vox_o[:-1]
    out[vox_o[first]] = lab_o[first]
    return out


def pca_features(xyz: np.ndarray, k: int = 16, chunk: int = 500_000, workers: int = -1) -> Dict[str, np.ndarray]:
    """Local covariance features from the k nearest neighbours (Weinmann et al. style):
    linearity, planarity, sphericity, verticality, plus unit normals (oriented to +z).
    float32 outputs; run on the voxel-subsampled cloud, not on raw 60 M points."""
    n = len(xyz)
    tree = cKDTree(xyz)
    lin = np.empty(n, np.float32); pla = np.empty(n, np.float32)
    sph = np.empty(n, np.float32); ver = np.empty(n, np.float32)
    normals = np.empty((n, 3), np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _, nb = tree.query(xyz[s:e], k=k, workers=workers)
        P = xyz[nb]                                   # (m, k, 3)
        P = P - P.mean(axis=1, keepdims=True)
        C = np.einsum("mki,mkj->mij", P, P) / k       # (m, 3, 3)
        w, V = np.linalg.eigh(C)                      # ascending eigenvalues
        w = np.clip(w, 1e-12, None)
        l3, l2, l1 = w[:, 0], w[:, 1], w[:, 2]
        lin[s:e] = (l1 - l2) / l1
        pla[s:e] = (l2 - l3) / l1
        sph[s:e] = l3 / l1
        nrm = V[:, :, 0]                              # eigenvector of the smallest eigenvalue
        flip = nrm[:, 2] < 0
        nrm[flip] *= -1
        normals[s:e] = nrm
        # verticality: 1 - |n_z| -> 1 for vertical surfaces/lines, 0 for horizontal ground
        ver[s:e] = 1.0 - np.abs(nrm[:, 2])
        print(f"\r[geometry] pca {e:,}/{n:,}", end="", flush=True)
    print()
    return {"linearity": lin, "planarity": pla, "sphericity": sph, "verticality": ver, "normals": normals}


def height_above_ground(xyz: np.ndarray, cell: float = 2.0, smooth: int = 2) -> np.ndarray:
    """Simple DTM: per-cell minimum z, filled and smoothed by a few rounds of 3×3 min/mean
    filtering, then HAG = z - DTM(x, y). Good enough for power-line corridors (open terrain);
    swap in CSF (pip install cloth-simulation-filter) later if needed."""
    origin = xyz[:, :2].min(axis=0)
    ij = np.floor((xyz[:, :2] - origin) / cell).astype(np.int64)
    W = ij[:, 0].max() + 1; H = ij[:, 1].max() + 1
    lin = ij[:, 1] * W + ij[:, 0]
    dtm = np.full(W * H, np.inf)
    np.minimum.at(dtm, lin, xyz[:, 2])
    dtm = dtm.reshape(H, W)
    # fill empty cells from neighbours, then light smoothing to suppress pits
    from scipy.ndimage import generic_filter, minimum_filter, uniform_filter
    empty = ~np.isfinite(dtm)
    if empty.any():
        filled = dtm.copy(); filled[empty] = np.nan
        for _ in range(50):
            m = np.isnan(filled)
            if not m.any():
                break
            nbmean = generic_filter(np.nan_to_num(filled, nan=0.0), np.sum, size=3, mode="nearest")
            nbcnt = generic_filter((~m).astype(float), np.sum, size=3, mode="nearest")
            with np.errstate(invalid="ignore", divide="ignore"):
                est = nbmean / nbcnt
            filled[m & (nbcnt > 0)] = est[m & (nbcnt > 0)]
        filled[np.isnan(filled)] = np.nanmin(filled)
        dtm = filled
    for _ in range(smooth):
        dtm = minimum_filter(dtm, size=3, mode="nearest")
    dtm = uniform_filter(dtm, size=3, mode="nearest")
    hag = xyz[:, 2] - dtm[ij[:, 1], ij[:, 0]]
    return np.maximum(hag, 0).astype(np.float32)
