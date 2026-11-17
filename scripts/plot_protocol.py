"""IoU-vs-pose-error curves from a protocol run (paper Fig. 2-3 draft).

  python scripts/plot_protocol.py --summary <out>/summary.json --out figures/protocol_fast_v1.png
  python scripts/plot_protocol.py --summary ... --classes conductor,structural,insulator,pylon --out figures/protocol_thin.png

Rows = class groups (default: mean over the four electrical classes, mean over all classes), columns = sweeps
(translation, rotation, drift); one line per model with a +-1 std band over the draws. The LiDAR-only reference is
flat by construction. A small table of the robustness score R and the drop at the reference level is printed.
"""
import argparse
import json

import numpy as np
import os as _os
_os.environ["MPLBACKEND"] = "Agg"     # a notebook kernel exports its inline backend; a plain subprocess cannot load it
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4df"
PALETTE = ["#6b6b6b", "#b3b3b3", "#2a78d6", "#eb6834", "#1f9d55", "#8e44ad", "#c0392b", "#d4a017"]
SWEEP_LABEL = {"trans": ("translation σ_t", "cm", 100.0), "rot": ("rotation σ_r", "°", 1.0),
               "drift": ("drift at end of line", "cm (and °/100)", 100.0),
               "trans_shared": ("shared translation σ_t", "cm", 100.0), "rot_shared": ("shared rotation σ_r", "°", 1.0)}
GROUP_LABEL = {"mean_electrical": "mean IoU, electrical classes", "mean_all": "mean IoU, all classes"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="figures/protocol.png")
    ap.add_argument("--classes", default="mean_electrical,mean_all", help="rows: class names or mean_electrical / mean_all")
    ap.add_argument("--models", default=None, help="comma list to restrict / order the models")
    ap.add_argument("--title", default=None)
    ap.add_argument("--tables", choices=["all", "compact"], default="all",
                    help="compact = only the one-row-per-model table (runs with many models)")
    a = ap.parse_args()
    s = json.load(open(a.summary))
    curves, scores, meta = s["curves"], s["scores"], s["meta"]
    models = a.models.split(",") if a.models else meta["models"]
    sweeps = [k for k in meta["sweeps"] if any(c["sweep"] == k for c in curves)]
    groups = a.classes.split(",")
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    style = {m: ("-", "--", ":")[(i // len(PALETTE)) % 3] for i, m in enumerate(models)}      # more models than colours

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(len(groups), len(sweeps), figsize=(4.2 * len(sweeps), 3.1 * len(groups)), dpi=200, squeeze=False)
    for gi, g in enumerate(groups):
        for si, sw in enumerate(sweeps):
            ax = axes[gi, si]
            label, unit, scale = SWEEP_LABEL.get(sw, (sw, "", 1.0))
            for m in models:
                pts = sorted([c for c in curves if c["model"] == m and c["sweep"] == sw and c["class"] == g], key=lambda c: c["level"])
                if not pts:
                    continue
                x = np.array([c["level"] for c in pts]) * scale
                y = np.array([c["iou_mean"] for c in pts]) * 100; e = np.array([c["iou_std"] for c in pts]) * 100
                ax.plot(x, y, style[m], marker="o", ms=3.5, lw=1.6, color=color[m], label=m)
                ax.fill_between(x, y - e, y + e, color=color[m], alpha=0.15, lw=0)
            ax.set_xlabel(f"{label} [{unit}]")
            if si == 0:
                ax.set_ylabel(f"{GROUP_LABEL.get(g, g + ' IoU')} [%]")
            ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if gi == 0 and si == 0:
                ax.legend(frameon=False, fontsize=8)
    title = a.title or f"Robustness protocol ({meta.get('mode')} mode, {meta.get('draws')} draws, zones: {', '.join(meta.get('zones', []))})"
    fig.suptitle(title, color=INK, fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out)

    # one row per model: mean IoU over all classes / over the electrical classes, clean -> reference -> largest level
    val = {(c["model"], c["sweep"], c["level"], c["class"]): c["iou_mean"] for c in curves}
    ref = meta.get("reference", {})
    cols = [(sweeps[0], 0.0, "clean")] if sweeps else []
    for sw in sweeps:
        levels = [float(x) for x in meta["sweeps"][sw]]
        lr = min(levels, key=lambda x: abs(x - ref.get(sw, levels[-1])))
        for lv in dict.fromkeys([lr, levels[-1]]):
            if lv > 0:
                cols.append((sw, lv, f"{sw} {lv:g}"))
    in_summary = [m for m in models if any(c["model"] == m for c in curves)]
    wm = max([len(m) for m in in_summary] + [5]) + 1
    wc = max([15] + [len(label) + 2 for _, _, label in cols])                  # long sweep names (trans_shared) need wider columns
    cell = lambda m, sw, lv: " / ".join("  -- " if (m, sw, lv, g) not in val or val[(m, sw, lv, g)] != val[(m, sw, lv, g)]
                                        else f"{100 * val[(m, sw, lv, g)]:5.1f}" for g in ("mean_all", "mean_electrical"))
    print("\nmean IoU [%]: all classes / electrical classes")
    print(f"{'model':{wm}s}" + "".join(f"{label:>{wc}s}" for _, _, label in cols))
    for m in in_summary:
        print(f"{m:{wm}s}" + "".join(f"{cell(m, sw, lv):>{wc}s}" for sw, lv, _ in cols))
    # the same columns in pixels: how far a projected point moves (median; a feature cell is 8 px, a probability cell 4 px)
    disp = {(d["sweep"], float(d["level"])): d for d in s.get("displacement", [])}
    if any(disp.get((sw, lv), {}).get("shift_px") is not None for sw, lv, _ in cols[1:]):
        px = lambda sw, lv, key: "--" if disp.get((sw, lv), {}).get(key) is None else f"{disp[(sw, lv)][key]:.1f}"
        print("\nmedian displacement of a projected point [px]: all points / points of the electrical classes")
        print(f"{'':{wm}s}" + "".join(f"{label:>{wc}s}" for _, _, label in cols))
        print(f"{'':{wm}s}" + "".join(f"{px(sw, lv, 'shift_px') + ' / ' + px(sw, lv, 'shift_px_electrical'):>{wc}s}" for sw, lv, _ in cols))
    if a.tables == "compact":
        return

    print(f"\n{'model':14s} {'sweep':6s} {'group':16s} {'clean':>6s} {'R':>6s} {'drop@ref':>9s} {'IoU@max':>8s}")
    for sc in scores:
        if sc["model"] in models and sc["class"] in groups:
            print(f"{sc['model']:14s} {sc['sweep']:6s} {sc['class']:16s} {100 * sc['clean_iou']:6.1f} {sc['R']:6.3f} "
                  f"{100 * sc['drop_ref']:9.1f} {100 * sc['iou_max_level']:8.1f}")

    # per class: clean IoU, then the IoU lost at the reference level and at the largest level of every sweep
    names = meta.get("class_names") or sorted({sc["class"] for sc in scores if not sc["class"].startswith("mean_")})
    by = {(sc["model"], sc["sweep"], sc["class"]): sc for sc in scores}
    fmt = lambda v: "   --" if v is None or v != v else f"{100 * v:5.1f}"
    for what, label in (("clean_iou", "clean IoU [%]"), ("drop_ref", "IoU lost at the reference level"), ("drop_max", "IoU lost at the largest level")):
        print(f"\n{label}" + ("" if what == "clean_iou" else "  (positive = worse than clean)"))
        print(f"{'model':14s} {'sweep':6s} {'level':>6s}  " + " ".join(f"{n[:9]:>9s}" for n in names))
        for m in models:
            for sw in sweeps:
                cells = []
                for n in names:
                    sc = by.get((m, sw, n))
                    v = None if sc is None else (sc["clean_iou"] - sc["iou_max_level"] if what == "drop_max" else sc[what])
                    cells.append(f"{fmt(v):>9s}")
                level = "" if what == "clean_iou" else (f"{ref.get(sw, float('nan')):g}" if what == "drop_ref" else f"{max(meta['sweeps'][sw]):g}")
                print(f"{m:14s} {sw:6s} {level:>6s}  " + " ".join(cells))
                if what == "clean_iou":
                    break                                   # clean numbers do not depend on the sweep


if __name__ == "__main__":
    main()
