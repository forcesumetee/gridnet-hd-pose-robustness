"""The two protocol figures of the paper, from the summary of the exact run (mean over the training seeds).

  python scripts/plot_paper_protocol.py                 # figures/protocol_benchmark.{pdf,png}, figures/protocol_fix.{pdf,png}
  python scripts/plot_paper_protocol.py --run lab3      # another protocol run with a summary in results/

Benchmark figure: LiDAR-only, ImageVote and the head types trained with the delivered poses.
Fix figure: every head type trained with the delivered poses (dashed, open markers) and with pose-noise augmentation
(solid, filled markers), with LiDAR-only as the reference line.
Rows = mean IoU over the electrical classes / over all classes; columns = translation, rotation, drift. A line is the
mean over the seeds of a head type (scripts/make_tables.py: FIX_GROUPS), its band the range between the seeds.
Head types differ in colour AND marker shape, the two kinds of training in line style AND marker fill.
"""
import argparse
import importlib.util
import json
import os

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_spec = importlib.util.spec_from_file_location("make_tables", os.path.join(ROOT, "scripts", "make_tables.py"))
mt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mt)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4df"
HEADS = [  # (head type of FIX_GROUPS, legend label, colour, marker): fixed order, colour follows the head type
    ("Late fusion", "Late fusion", "#2a78d6", "o"),
    ("Feature concat.", "Feature concatenation", "#eb6834", "s"),
    ("Window $3{\\times}3$", "Window 3×3", "#1f9d55", "^"),
    ("Window $5{\\times}5$", "Window 5×5", "#8e44ad", "D"),
]
REF = {"lidar": ("LiDAR-only", "#52514e", ":"), "imagevote": ("ImageVote", "#9a9a96", "-")}
SWEEPS = [("trans", "translation error $\\sigma_t$ [cm]", 100.0), ("rot", "rotation error $\\sigma_r$ [°]", 1.0),
          ("drift", "drift at the end of the line [cm]", 100.0)]
ROWS = [("mean_electrical", "mean IoU [%]\nelectrical classes"), ("mean_all", "mean IoU [%]\nall classes")]   # two lines: a panel is 1.3 in high
FIG_SIZE = (7.2, 3.7)          # inches; the paper prints the figure 7.0 in wide (4.5 in high until update50: the two
#                                figures took 47 % of a page each)
LEGEND_ROW = 0.17              # inches per legend row, incl. the gap to the panels


def group_curve(val, names, sweep, cls, levels):
    """(mean, min, max) per level over the heads of `names` that have the curve; None when no head has it."""
    names = [n for n in names if (n, sweep, levels[0], cls) in val]
    if not names:
        return None
    ys = [[val[(n, sweep, lv, cls)] for n in names] for lv in levels]
    return [sum(y) / len(y) for y in ys], [min(y) for y in ys], [max(y) for y in ys]


def draw(summary, kind, out_base):
    """kind: "benchmark" or "fix". Writes <out_base>.pdf and <out_base>.png; returns the legend labels."""
    val, _ = mt.merged_curves([("run", summary)])
    meta = summary["meta"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#b8b7b2", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(ROWS), len(SWEEPS), figsize=FIG_SIZE, sharey="row", squeeze=False)
    handles = []
    for ri, (cls, ylabel) in enumerate(ROWS):
        for ci, (sw, xlabel, scale) in enumerate(SWEEPS):
            ax = axes[ri, ci]
            levels = [float(x) for x in meta["sweeps"][sw]]
            x = [lv * scale for lv in levels]
            refs = ["lidar", "imagevote"] if kind == "benchmark" else ["lidar"]
            for key in refs:
                c = group_curve(val, [key], sw, cls, levels)
                if c:
                    label, color, ls = REF[key]
                    ax.plot(x, c[0], ls, color=color, lw=1.4, marker="x" if key == "imagevote" else None, ms=4, zorder=2)
                    if ri == 0 and ci == 0:
                        handles.append(Line2D([], [], color=color, ls=ls, lw=1.4, marker="x" if key == "imagevote" else None, ms=4, label=label))
            for label, legend, color, marker in HEADS:
                for block, ls, filled in (("Delivered", "--" if kind == "fix" else "-", kind != "fix"), ("Pose-noise", "-", True)):
                    if kind == "benchmark" and block != "Delivered":
                        continue
                    c = group_curve(val, mt._group(label, block), sw, cls, levels)
                    if not c:
                        continue
                    face = color if filled else "white"
                    ax.fill_between(x, c[1], c[2], color=color, alpha=0.16, lw=0, zorder=1)
                    ax.plot(x, c[0], ls, color=color, lw=1.4, marker=marker, ms=4.2, mfc=face, mec=color, mew=1.0, zorder=3)
                    if ri == 0 and ci == 0:
                        name = legend if kind == "benchmark" else f"{legend}, {'pose noise' if block == 'Pose-noise' else 'delivered poses'}"
                        handles.append(Line2D([], [], color=color, ls=ls, lw=1.4, marker=marker, ms=4.2, mfc=face, mec=color, mew=1.0, label=name))
            ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}" if (i != 1 or sw == "drift") else "" for i, v in enumerate(x)])   # 0 and the first level overlap
            if ri == len(ROWS) - 1:
                ax.set_xlabel(xlabel)
            if ci == 0:
                ax.set_ylabel(ylabel)
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.6)
    rows = -(-len(handles) // 3)                                # legend rows at three columns
    fig.tight_layout(rect=(0, 0, 1, 1 - (rows * LEGEND_ROW + 0.06) / FIG_SIZE[1]), h_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=220)
    plt.close(fig)
    return [h.get_label() for h in handles]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=mt.EXACT_RUN, help="protocol run with results/protocol_<run>_summary.json")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()
    summary = json.load(open(os.path.join(ROOT, "results", f"protocol_{a.run}_summary.json")))
    for kind in ("benchmark", "fix"):
        base = os.path.join(a.out_dir, f"protocol_{kind}")
        labels = draw(summary, kind, base)
        print(f"wrote {base}.pdf / .png  ({len(labels)} lines: {', '.join(labels)})")


if __name__ == "__main__":
    main()
