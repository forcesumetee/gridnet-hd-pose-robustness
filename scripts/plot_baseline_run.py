"""Training-run figure for the LiDAR-only baseline (loss curve, val mIoU per eval, per-class IoU).

  python scripts/plot_baseline_run.py --log data/colab_logs/litept_s_lidar_v2.txt --out figures/baseline_v2_training.png

The log is the cell-6 output of colab_train_litept.ipynb (one or more sessions concatenated).
Per-class IoU is taken from the eval line of the best checkpoint (highest mIoU) in the log.
"""
import argparse
import re

import os as _os
_os.environ["MPLBACKEND"] = "Agg"     # a notebook kernel exports its inline backend; a plain subprocess cannot load it
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb", "rock_soil", "road", "water", "building"]
ELECTRICAL = {"pylon", "conductor", "structural", "insulator"}
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e5e4df"


def parse(path):
    its, losses, evals = [], [], []            # evals: (it, mIoU, OA, {cls: iou})
    lines = open(path, encoding="utf-8").read().splitlines()
    for i, l in enumerate(lines):
        m = re.match(r"\s*it\s+(\d+)/\d+\s+loss\s+([\d.]+|nan)", l)
        if m and m.group(2) != "nan":
            its.append(int(m.group(1))); losses.append(float(m.group(2)))
        m = re.match(r"\[eval\] it (\d+)\s+mIoU ([\d.]+)\s+OA ([\d.]+)", l)
        if m:
            per = dict(re.findall(r"(\w+) ([\d.]+)", lines[i + 1])) if i + 1 < len(lines) else {}
            evals.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), {k: float(v) for k, v in per.items()}))
    # a resumed session repeats earlier lines in the pasted log: keep the last occurrence per iteration
    d = {}
    for it, lo in zip(its, losses):
        d[it] = lo
    its = sorted(d); losses = [d[i] for i in its]
    ev = {}
    for e in evals:
        ev[e[0]] = e
    return its, losses, [ev[k] for k in sorted(ev)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="figures/baseline_v2_training.png")
    ap.add_argument("--ref", type=float, default=72.88, help="leaderboard entry Lite-S-test (hidden test; no publication) for reference")
    ap.add_argument("--title", default="LitePT-S LiDAR-only baseline on GridNet-HD (run v2, A100-40GB, 20k its, 7 h)")
    a = ap.parse_args()
    its, losses, evals = parse(a.log)
    best = max(evals, key=lambda e: e[1])

    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.15, 0.9, 1.25]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axs:
        ax.set_facecolor("#fcfcfb"); ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # 1) training loss (50-it means as logged) + a light running mean
    ax = axs[0]
    ax.plot(its, losses, color=BLUE, linewidth=1.0, alpha=0.35)
    k = 10
    if len(losses) > k:
        run = [sum(losses[max(0, i - k + 1):i + 1]) / len(losses[max(0, i - k + 1):i + 1]) for i in range(len(losses))]
        ax.plot(its, run, color=BLUE, linewidth=2.0)
    ax.set_xlabel("iteration"); ax.set_ylabel("training loss (CE + Lovász)")
    ax.set_title("Training loss", loc="left", fontsize=12, color=INK)
    ax.set_xlim(0, max(its)); ax.set_ylim(0, max(losses) * 1.05)
    ax.annotate(f"{losses[-1]:.2f}", xy=(its[-1], losses[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=10)

    # 2) val mIoU per evaluation
    ax = axs[1]
    xs = [e[0] for e in evals]; ys = [e[1] for e in evals]
    ax.plot(xs, ys, color=ORANGE, linewidth=2.0, marker="o", markersize=7, markerfacecolor="#fcfcfb", markeredgewidth=2)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 9), textcoords="offset points", ha="center", color=INK2, fontsize=10)
    ax.axhline(a.ref, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"leaderboard entry Lite-S-test (hidden test) {a.ref:.1f}", xy=(min(xs), a.ref), xytext=(0, 5), textcoords="offset points",
                color=INK2, fontsize=9.5)
    ax.set_xlabel("iteration"); ax.set_ylabel("val mIoU (6 zones)")
    ax.set_title("Validation mIoU", loc="left", fontsize=12, color=INK)
    ax.set_ylim(min(ys) - 6, max(max(ys), a.ref) + 5); ax.set_xlim(min(xs) - 1500, max(xs) + 1500)

    # 3) per-class IoU at the best checkpoint
    ax = axs[2]
    per = best[3]
    names = [c for c in CLASSES if c in per]
    vals = [per[c] for c in names]
    cols = [BLUE if c in ELECTRICAL else ORANGE for c in names]
    y = list(range(len(names)))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.1f}", xy=(v, yi), xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.set_xlim(0, 112); ax.set_xlabel("IoU (%)")
    ax.set_title(f"Per-class IoU at best checkpoint (it {best[0]}: mIoU {best[1]:.1f}, OA {best[2]:.1f})", loc="left", fontsize=12, color=INK)
    ax.grid(True, axis="x", color=GRID); ax.grid(False, axis="y")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="power-line structures (thin classes)"), Patch(color=ORANGE, label="vegetation / ground / building")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9.5)

    fig.suptitle(a.title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(a.out, dpi=200, facecolor=fig.get_facecolor())
    print("wrote", a.out, "| evals:", [(e[0], e[1]) for e in evals], "| best it", best[0])


if __name__ == "__main__":
    main()
