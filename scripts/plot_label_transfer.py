"""Figure 1 of the paper: model-free label transfer under camera-pose error, as a vector figure.

  python scripts/plot_label_transfer.py            # results/perturb_t5b5.csv -> figures/label_transfer_t5b5.pdf + .png
  python scripts/plot_label_transfer.py --csv results/perturb_t5b5.csv --out figures/label_transfer_t5b5
  python scripts/plot_label_transfer.py --csv <out>/all/label_transfer.csv --out figures/label_transfer_val

Two sources. (1) `scripts/demo_ab_project.py perturb` (columns perturbation, level, all, conductor, structural,
insulator, n): the points seen in ONE image of a zone receive the label of the pixel they project to; the camera of
that image is displaced by a fixed distance in a random direction, or rotated by a fixed angle about a random axis,
and every level is the mean over the trials (5). `all` is the share of all visible points that receive their own
label, the class columns the same share among the points of that class.
(2) `scripts/label_transfer_zone.py` (all images of one or several zones; gridnet/label_transfer.py): the same
shares pooled over the images, a column `noise` (protocol = sigma per axis, the error model of the robustness
protocol; fixed = as in (1)) and the quartiles over the images (`<column>_q25`, `<column>_q75`), drawn as a band.

Two panels on one y-axis (position error | rotation error); the levels are equally spaced because they grow roughly
geometrically (2, 5, 10, 25, 50 cm). Colours: the all-class reference in grey, the three thin classes in three hues
that stay apart under colour-vision deficiency (checked with the dataviz validator, all pairs) plus one marker shape
each, so the figure also reads in greyscale.
"""
import argparse
import csv
import os

os.environ["MPLBACKEND"] = "Agg"     # a notebook kernel exports its inline backend; a plain subprocess cannot load it
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e5e4df", "#c3c2b7"
SERIES = [  # (csv column, legend label, colour, marker)
    ("all", "All classes", "#6b6b6b", "o"),
    ("conductor", "Conductor cable", "#2a78d6", "s"),
    ("structural", "Structural cable", "#eb6834", "D"),
    ("insulator", "Insulator", "#1baf7a", "^"),
]
# (csv value of `perturbation`, x label per error model, reference level, where the value label of the structural cable sits)
PANELS = [("translation_cm", {"fixed": "(a) Camera position error [cm]", "protocol": "(a) Camera position error $\\sigma_t$ [cm]"},
           10.0, ((-7, -3), "right")),
          ("rotation_deg", {"fixed": "(b) Camera rotation error [°]", "protocol": "(b) Camera rotation error $\\sigma_r$ [°]"},
           0.1, ((7, 4), "left"))]
CALLOUT = "structural"            # the one series whose value is written out, at the reference level of each panel


def load(path):
    """{kind: [(level, {column: percent, column_q25: percent, ...})]} with the unperturbed row first in every kind,
    and the error model of the file ("fixed" for the single-image CSV, which has no `noise` column)."""
    rows = list(csv.DictReader(open(path, newline="")))
    cols = [s[0] for s in SERIES]
    cols += [f"{c}_{q}" for c in cols for q in ("q25", "q75") if f"{c}_{q}" in rows[0]]
    base = [r for r in rows if r["perturbation"] == "none"]
    out = {}
    for kind, *_ in PANELS:
        sel = base + sorted((r for r in rows if r["perturbation"] == kind), key=lambda r: float(r["level"]))
        out[kind] = [(float(r["level"]), {c: 100 * float(r[c]) for c in cols}) for r in sel]
    return out, rows[0].get("noise", "fixed")


def draw(data, out_base, noise="fixed"):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.3, 2.75), sharey=True)
    for ax, (kind, xlabels, ref, ((dx, dy), ha)) in zip(axes, PANELS):
        pts = data[kind]
        xs = list(range(len(pts)))
        for col, label, color, marker in SERIES:
            if f"{col}_q25" in pts[0][1]:                  # spread over the images: interquartile band, behind the lines
                ax.fill_between(xs, [v[f"{col}_q25"] for _, v in pts], [v[f"{col}_q75"] for _, v in pts], color=color,
                                alpha=0.13, lw=0, zorder=2)
            ax.plot(xs, [v[col] for _, v in pts], color=color, lw=1.5, marker=marker, ms=5.5 if marker != "D" else 4.8,
                    mec="white", mew=0.9, solid_capstyle="round", solid_joinstyle="round", label=label, clip_on=False,
                    zorder=2.5 if col == "all" else (3.5 if col == CALLOUT else 3))
        i_ref = min(xs, key=lambda i: abs(pts[i][0] - ref))
        v = pts[i_ref][1][CALLOUT]
        if v == v:                                         # NaN when the class does not occur
            ax.annotate(f"{v:.0f} %", (i_ref, v), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                        fontsize=7.5, color=INK, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lv:g}" for lv, _ in pts])
        ax.set_xlabel(xlabels[noise])
        ax.set_xlim(-0.25, len(xs) - 0.75)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=4)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Points receiving their own label [%]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(SERIES), frameon=False, fontsize=8, handlelength=2.4,
               columnspacing=1.8, bbox_to_anchor=(0.53, 1.0), labelcolor=INK)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.17, wspace=0.07)
    os.makedirs(os.path.dirname(os.path.abspath(out_base)), exist_ok=True)
    paths = [out_base + ".pdf", out_base + ".png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "perturb_t5b5.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "label_transfer_t5b5"), help="output path without extension")
    a = ap.parse_args()
    data, noise = load(a.csv)
    for p in draw(data, a.out, noise):
        print("wrote", p)
    print("error model:", noise)
    for kind, _, ref, _ in PANELS:                   # the numbers the text of the paper quotes
        pts = data[kind]
        r = min(pts, key=lambda p: abs(p[0] - ref))
        print(f"{kind}: clean " + ", ".join(f"{c} {pts[0][1][c]:.1f}" for c, *_ in SERIES)
              + f" | at {ref:g}: " + ", ".join(f"{c} {r[1][c]:.1f}" for c, *_ in SERIES)
              + f" | at {pts[-1][0]:g}: " + ", ".join(f"{c} {pts[-1][1][c]:.1f}" for c, *_ in SERIES))


if __name__ == "__main__":
    main()
