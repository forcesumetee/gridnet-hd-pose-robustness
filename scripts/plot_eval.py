"""Validation analysis figure from eval.json: row-normalised confusion matrix + mIoU per val zone.

  python scripts/plot_eval.py --eval <ckpt_dir>/eval.json --out figures/eval_val.png

eval.json is written by `scripts/train_litept.py --eval-only` (needs the version with
"confusion" and "per_zone" keys). Rows of the matrix are ground truth, columns predictions,
each row normalised to 100 % (recall per class on the diagonal).
"""
import argparse
import json
import os

import numpy as np
import os as _os
_os.environ["MPLBACKEND"] = "Agg"     # a notebook kernel exports its inline backend; a plain subprocess cannot load it
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

LABELS = ["Pylon", "Conductor", "Struct. cable", "Insulator", "High veg.", "Low veg.", "Herbaceous",
          "Rock / soil", "Road", "Water", "Building"]
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e5e4df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default="figures/eval_val.png")
    ap.add_argument("--title", default="LitePT-S LiDAR-only baseline — validation zones")
    a = ap.parse_args()
    e = json.load(open(a.eval))
    cm = np.asarray(e["confusion"], dtype=np.float64)
    row = cm.sum(1, keepdims=True)
    pct = 100 * cm / np.maximum(row, 1)
    share = 100 * row[:, 0] / cm.sum()                       # share of labelled points per class

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(12.5, 5.6), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.3, left=0.12, right=0.98, top=0.86, bottom=0.2)

    # --- confusion matrix (sequential single hue: white -> blue)
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#d6e6f8", BLUE, "#123c73"])
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100, aspect="equal")
    n = len(LABELS)
    for i in range(n):
        for j in range(n):
            v = pct[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.2,
                        color="white" if v > 55 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"{l}  ({s:.1f}%)" if s >= 0.1 else f"{l}  ({s:.2f}%)" for l, s in zip(LABELS, share)], fontsize=8)
    ax.set_xlabel("predicted", color=INK2); ax.set_ylabel("ground truth  (share of points)", color=INK2)
    ax.set_title("Confusion matrix, % of each ground-truth class", fontsize=10, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=GRID, lw=0.6); ax.tick_params(which="minor", length=0)

    # --- mIoU per zone (single series, no legend)
    ax2 = fig.add_subplot(gs[1])
    zones = list(e["per_zone"].keys())
    mi = [e["per_zone"][z]["mIoU"] for z in zones]
    oa = [e["per_zone"][z]["OA"] for z in zones]
    x = np.arange(len(zones))
    ax2.bar(x, mi, width=0.62, color=BLUE, zorder=3)
    ax2.axhline(e["mIoU"], color=ORANGE, lw=1.4, zorder=4)
    ax2.text(len(zones) - 0.5, e["mIoU"] + 1.2, f"all zones {e['mIoU']:.1f}", ha="right", va="bottom", fontsize=8.5, color=ORANGE)
    for xi, m, o in zip(x, mi, oa):
        ax2.text(xi, m - 4.5, f"{m:.1f}", ha="center", va="top", fontsize=8.5, color="white", fontweight="bold")
        ax2.text(xi, 2, f"OA {o:.0f}", ha="center", va="bottom", fontsize=7.2, color="white")
    ax2.set_xticks(x); ax2.set_xticklabels(zones, fontsize=8.5)
    ax2.set_ylim(0, 100); ax2.set_ylabel("mIoU (%)", color=INK2)
    ax2.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_title("mIoU per validation zone", fontsize=10, color=INK, loc="left")

    ck = f"  ·  checkpoint it {e['it']:,}" if e.get("it") else ""
    fig.suptitle(f"{a.title}{ck}  ·  mIoU {e['mIoU']:.1f}  OA {e['OA']:.1f}",
                 fontsize=11, color=INK, x=0.12, ha="left", y=0.965)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
