"""Qualitative figures from saved predictions: a top-down view of the whole zone and a side view
through one pylon, each as ground truth | prediction | errors.

  python scripts/render_predictions.py --cache /content/cache --zone t5b5 \
      --pred <ckpt_dir>/pred/t5b5_pred.npy --out figures/qual_t5b5
  -> figures/qual_t5b5_topdown.png, figures/qual_t5b5_side.png

Predictions come from `scripts/train_litept.py --eval-only --save-pred <dir>` (uint8 per cache
point, same order as points.npz). Pure numpy + matplotlib; no torch needed. Rendering is a
"z-buffer" raster: each cell shows the point nearest to the viewer (highest point for the top-down
view, nearest point of a slab for the side view), so thin structures stay visible.
"""
import argparse
import os
import sys

import numpy as np
import os as _os
_os.environ["MPLBACKEND"] = "Agg"     # a notebook kernel exports its inline backend; a plain subprocess cannot load it
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gridnet.classes import CLASSES, PALETTE  # noqa: E402

INK, INK2 = "#0b0b0b", "#52514e"
OK_RGB, ERR_RGB, UNLAB_RGB = (226, 226, 222), (214, 32, 32), (250, 250, 250)


def raster(u, v, depth, cell):
    """Bin (u, v) into cells of size `cell`; per cell keep the point with the largest `depth`.
    Returns (rows, cols, sel, H, W): the cell of each kept point and its index into the input."""
    u0, v0 = u.min(), v.min()
    col = np.floor((u - u0) / cell).astype(np.int64)
    row = np.floor((v - v0) / cell).astype(np.int64)
    W, H = int(col.max()) + 1, int(row.max()) + 1
    key = row * W + col
    order = np.lexsort((depth, key))                 # by cell, then depth ascending
    k = key[order]
    last = np.empty(len(k), bool)
    last[:-1] = k[1:] != k[:-1]
    last[-1] = True
    sel = order[last]
    return row[sel], col[sel], sel, H, W


def images(rows, cols, H, W, gt, pred):
    """RGB images (origin at the bottom-left) for ground truth, prediction and errors."""
    def paint(lab):
        img = np.full((H, W, 3), 255, np.uint8)
        img[rows, cols] = PALETTE[lab]
        return img[::-1]                             # v grows upwards
    err = np.full((H, W, 3), 255, np.uint8)
    col = np.where((gt == pred)[:, None], np.array(OK_RGB, np.uint8), np.array(ERR_RGB, np.uint8))
    col[gt == 255] = UNLAB_RGB
    err[rows, cols] = col
    # dilate wrong cells by one cell so 1-px errors on cables stay visible at figure scale
    wrong = np.zeros((H, W), bool)
    wrong[rows, cols] = (gt != pred) & (gt != 255)
    d = wrong.copy()
    d[1:] |= wrong[:-1]; d[:-1] |= wrong[1:]; d[:, 1:] |= wrong[:, :-1]; d[:, :-1] |= wrong[:, 1:]
    err[d] = ERR_RGB
    lab = gt != 255
    acc = float((gt[lab] == pred[lab]).mean()) if lab.any() else float("nan")
    return paint(gt), paint(pred), err[::-1], acc


def panel_figure(imgs, titles, extent, out, suptitle, legend_classes, unit="m"):
    H, W = imgs[0].shape[:2]
    wide = W / max(H, 1) > 1.6
    n = len(imgs)
    if wide:
        fig, axes = plt.subplots(n, 1, figsize=(11, 1.2 + n * (11 * H / W + 0.5)), dpi=150)
    else:
        fig, axes = plt.subplots(1, n, figsize=(1.5 + n * 5.2, 5.2 * H / W + 1.6), dpi=150)
    for ax, img, t in zip(np.atleast_1d(axes), imgs, titles):
        ax.imshow(img, extent=extent, interpolation="nearest")
        ax.set_title(t, fontsize=10, color=INK, loc="left")
        ax.tick_params(labelsize=7, colors=INK2, length=2)
        for s in ax.spines.values():
            s.set_color("#d0d0cc")
    handles = [Patch(facecolor=PALETTE[c] / 255, edgecolor="none", label=CLASSES[c]) for c in legend_classes]
    handles += [Patch(facecolor=np.array(OK_RGB) / 255, label="correct"), Patch(facecolor=np.array(ERR_RGB) / 255, label="wrong")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(suptitle, fontsize=11, color=INK, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.06 if wide else 0.12, 1, 0.975))
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


def densest_pylon(xyz, labels, bin_m=10.0):
    """Centre (x, y) of the pylon with the most points, from a coarse 2D histogram of pylon points."""
    p = xyz[labels == 0, :2]
    if len(p) < 500:
        return None
    lo = p.min(0)
    ij = np.floor((p - lo) / bin_m).astype(np.int64)
    nx, ny = ij[:, 0].max() + 1, ij[:, 1].max() + 1
    h = np.bincount(ij[:, 0] * ny + ij[:, 1], minlength=nx * ny).reshape(nx, ny)
    i, j = np.unravel_index(h.argmax(), h.shape)
    near = (np.abs(ij[:, 0] - i) <= 1) & (np.abs(ij[:, 1] - j) <= 1)
    return p[near].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--pred", required=True, help="<zone>_pred.npy from --save-pred")
    ap.add_argument("--out", required=True, help="output prefix, e.g. figures/qual_t5b5")
    ap.add_argument("--cell", type=float, default=0.25, help="top-down cell size (m)")
    ap.add_argument("--side-cell", type=float, default=0.05, help="side-view cell size (m)")
    ap.add_argument("--side-radius", type=float, default=35.0, help="side-view crop radius around the pylon (m)")
    ap.add_argument("--slab", type=float, default=12.0, help="side view keeps |depth| <= slab (m) around the pylon")
    ap.add_argument("--centre", default=None, help="x,y of the side-view centre (zone-relative m); default = densest pylon")
    ap.add_argument("--no-side", action="store_true")
    a = ap.parse_args()

    p = np.load(os.path.join(a.cache, a.zone, "points.npz"))
    xyz = p["xyz"].astype(np.float32)
    gt = p["labels"].astype(np.uint8)
    pred = np.load(a.pred).astype(np.uint8)
    assert len(pred) == len(gt), f"prediction has {len(pred)} points, cache {len(gt)}"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    present = [int(c) for c in np.unique(gt) if c != 255]

    # ---- top-down: highest point per cell
    rows, cols, sel, H, W = raster(xyz[:, 0], xyz[:, 1], xyz[:, 2], a.cell)
    g, q, e, acc = images(rows, cols, H, W, gt[sel], pred[sel])
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    ext = (x0, x0 + W * a.cell, y0, y0 + H * a.cell)
    panel_figure([g, q, e], ["Ground truth", "Prediction", f"Errors ({100 * (1 - acc):.1f} % of rendered points wrong)"],
                 ext, a.out + "_topdown.png",
                 f"Zone {a.zone} — top-down view, {a.cell:g} m cells, {len(gt) / 1e6:.0f} M points", present)

    if a.no_side:
        return
    c = np.array([float(v) for v in a.centre.split(",")]) if a.centre else densest_pylon(xyz, gt)
    if c is None:
        print("no pylon in this zone; side view skipped")
        return
    d2 = ((xyz[:, :2] - c) ** 2).sum(1)
    crop = np.where(d2 <= a.side_radius ** 2)[0]
    cx = xyz[crop]
    cond = cx[gt[crop] == 1, :2]
    src = cond if len(cond) >= 100 else cx[:, :2]
    _, _, vt = np.linalg.svd(src - src.mean(0), full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])              # along the conductors -> horizontal axis of the view
    nrm = np.array([-d[1], d[0]])
    rel = cx[:, :2] - c
    s, t = rel @ d, rel @ nrm
    keep = np.abs(t) <= a.slab
    s, t, z = s[keep], t[keep], cx[keep, 2]
    sub = crop[keep]
    rows, cols, sel, H, W = raster(s, z, -t, a.side_cell)   # nearest to a viewer on the -t side
    g, q, e, acc = images(rows, cols, H, W, gt[sub][sel], pred[sub][sel])
    ext = (s.min(), s.min() + W * a.side_cell, z.min(), z.min() + H * a.side_cell)
    panel_figure([g, q, e], ["Ground truth", "Prediction", f"Errors ({100 * (1 - acc):.1f} % of rendered points wrong)"],
                 ext, a.out + "_side.png",
                 f"Zone {a.zone} — side view through the pylon at ({c[0]:.0f}, {c[1]:.0f}) m, "
                 f"{a.side_cell:g} m cells, slab ±{a.slab:g} m", present)


if __name__ == "__main__":
    main()
