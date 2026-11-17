"""Figure of the LiDAR-only reference for the paper, composed from two renderings of scripts/render_predictions.py.

  python scripts/compose_reference_figure.py
      [--side figures/qual_t4z1_side.png] [--top figures/qual_t5b5_topdown.png] [--out figures/qualitative_reference]
  -> figures/qualitative_reference.{pdf,png}

The renderings are figures for a screen: three panels with axis ticks in metres and a legend, 17 inches wide. Scaled to
a page of the paper their lettering is 2-3 pt. This script cuts the three panels out of each rendering (the pixels
inside the axes frames, unchanged), sets the six panels next to each other and writes titles, scale bars and one legend
at the size of the paper. Nothing is rendered again, so the point clouds are not needed: the frames are found by their
colour, and the scale bars are derived from the tick marks of the renderings, which are TICK_STEP metres apart.
numpy + matplotlib, no torch.
"""
import argparse
import os
import sys

import numpy as np

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from gridnet.classes import CLASSES, PALETTE  # noqa: E402

INK, INK2, EDGE = "#0b0b0b", "#52514e", "#c9c8c3"
SPINE_RGB = (208, 208, 204)                  # the axes frame of render_predictions.panel_figure ("#d0d0cc")
OK_RGB, ERR_RGB = (226, 226, 222), (214, 32, 32)     # render_predictions: a correct and a wrong point in the error panel
# what the two renderings of the paper show (read off their titles, tick labels and legends)
SIDE = {"file": "qual_t4z1_side.png", "label": "(a) Zone t4z1, side view through a pylon", "tick_step": 10.0, "bar": 10.0}
TOP = {"file": "qual_t5b5_topdown.png", "label": "(b) Zone t5b5, top-down view", "tick_step": 50.0, "bar": 100.0}
TITLES = ("Ground truth", "Prediction", "Errors")
LEGEND_CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)     # the classes in the legends of the two renderings (no water)
NAMES = {8: "Road"}                                  # the name of the paper where it differs from the dataset's
FIG_WIDTH, TOP_HEIGHT = 7.0, 1.72                    # inches: text width of the paper; height of a top-down panel


def load_rgb(path):
    """A PNG as uint8 RGB."""
    a = mpimg.imread(path)
    if a.dtype != np.uint8:
        a = np.round(a * 255).astype(np.uint8)
    return a[..., :3]


def _runs(index):
    """[(first, last)] of the runs of consecutive integers in a sorted index array."""
    if len(index) == 0:
        return []
    cut = np.flatnonzero(np.diff(index) > 1)
    first = np.concatenate([[0], cut + 1])
    last = np.concatenate([cut, [len(index) - 1]])
    return [(int(index[a]), int(index[b])) for a, b in zip(first, last)]


def spine_mask(rgb):
    """Pixels in the colour of an axes frame of render_predictions.panel_figure: its light grey ("#d0d0cc") at a coverage
    of 70 % or more over white, whatever the anti-aliasing of the matplotlib that drew it. The grey of a correct point
    in the error panel (OK_RGB) is lighter and stays outside."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return (r >= SPINE_RGB[0] - 4) & (r <= SPINE_RGB[0] + 14) & (np.abs(r - g) <= 2) & (r - b >= 1) & (r - b <= 6)


def frames(rgb, share=0.4):
    """The axes frames of a rendering with its panels in one row: [(x0, x1, y0, y1)], the pixel ranges INSIDE each frame
    (x1 and y1 exclusive), from left to right. A frame line is a row or a column in which at least `share` of the
    pixels have the frame colour."""
    spine = spine_mask(rgb)
    cols = _runs(np.flatnonzero(spine.sum(0) >= share * rgb.shape[0]))
    rows = _runs(np.flatnonzero(spine.sum(1) >= share * rgb.shape[1]))
    if len(rows) != 2 or len(cols) < 2 or len(cols) % 2:
        raise ValueError(f"expected 2 horizontal and an even number of vertical frame lines, found {len(rows)} and {len(cols)}")
    y0, y1 = rows[0][1] + 1, rows[1][0]
    return [(left[1] + 1, right[0], y0, y1) for left, right in zip(cols[0::2], cols[1::2])]


def tick_spacing(rgb, frame, dark=200):
    """Distance in pixels between neighbouring tick marks under a frame (first to last, divided by the gaps between)."""
    x0, x1, _, y1 = frame
    spine = spine_mask(rgb[:, x0:x1])
    row = y1
    while row < rgb.shape[0] and spine[row].mean() > 0.5:
        row += 1                                         # past the frame line: the first row that holds only tick marks
    line = (rgb[row, max(x0 - 4, 0):x1 + 4] < dark).all(-1)
    ticks = [0.5 * (a + b) for a, b in _runs(np.flatnonzero(line))]
    if len(ticks) < 3:
        raise ValueError(f"found {len(ticks)} tick marks under the frame")
    gaps = np.diff(ticks)
    if gaps.max() - gaps.min() > 2:                      # a pixel of rounding per tick, not a missing tick
        raise ValueError(f"the tick marks under the frame are not evenly spaced: {gaps.tolist()}")
    return float((ticks[-1] - ticks[0]) / (len(ticks) - 1))


def panels(path, tick_step):
    """(crops, metres per pixel) of a rendering: the pixels inside its frames (cut to a common size), and its scale from
    the tick marks."""
    rgb = load_rgb(path)
    fr = frames(rgb)
    widths, heights = [x1 - x0 for x0, x1, _, _ in fr], [y1 - y0 for _, _, y0, y1 in fr]
    if max(widths) - min(widths) > 2 or max(heights) - min(heights) > 2:     # frames snap to pixels: one pixel is normal
        raise ValueError(f"the panels of {path} differ in size: {sorted(set(zip(widths, heights)))}")
    w, h = min(widths), min(heights)
    return [rgb[y0:y0 + h, x0:x0 + w] for x0, _, y0, _ in fr], tick_step / tick_spacing(rgb, fr[0])


def layout(aspect_side, aspect_top, width=FIG_WIDTH, top_height=TOP_HEIGHT, gap=0.05, block_gap=0.16, head=0.16, foot=0.17):
    """Geometry in inches: side-view panels on the left, top-down panels on the right, all in one figure of `width`.
    Returns (figure height, side panel (w, h), top panel (w, h), x of the first top-down panel)."""
    tw = aspect_top * top_height
    right = 3 * tw + 2 * gap
    sw = (width - right - block_gap - 2 * gap) / 3
    if sw <= 0:
        raise ValueError("no room for the side view")
    return head + top_height + foot, (sw, sw / aspect_side), (tw, top_height), width - right


def scale_bar(ax, shape, m_per_px, metres, corner):
    """A bar of `metres` in a corner of a panel that is empty in the rendering ("ll" or "lr")."""
    h, w = shape[:2]
    length = metres / m_per_px
    y = h - 0.045 * h if h < w else h - 0.02 * h
    x = 0.04 * w if corner == "ll" else w - 0.06 * w - length
    ax.plot([x, x + length], [y, y], color=INK, lw=1.0, solid_capstyle="butt")
    ax.text(x + 0.5 * length, y - 0.012 * max(h, w), f"{metres:g} m", ha="center", va="bottom", fontsize=6, color=INK)


def compose(side_png, top_png, out_base, dpi=300, formats=("pdf", "png")):
    side, side_scale = panels(side_png, SIDE["tick_step"])
    top, top_scale = panels(top_png, TOP["tick_step"])
    if len(side) != 3 or len(top) != 3:
        raise ValueError(f"expected three panels per rendering, found {len(side)} and {len(top)}")
    gap, head, foot = 0.05, 0.16, 0.17
    height, (sw, sh), (tw, th), x_top = layout(side[0].shape[1] / side[0].shape[0], top[0].shape[1] / top[0].shape[0],
                                                gap=gap, head=head, foot=foot)
    fig = plt.figure(figsize=(FIG_WIDTH, height))

    def block(crops, x, w, h, m_per_px, spec, corner):
        y = height - head - h
        for i, (crop, title) in enumerate(zip(crops, TITLES)):
            ax = fig.add_axes([(x + i * (w + gap)) / FIG_WIDTH, y / height, w / FIG_WIDTH, h / height])
            ax.imshow(crop, interpolation="none", aspect="auto")
            ax.set_xlim(-0.5, crop.shape[1] - 0.5); ax.set_ylim(crop.shape[0] - 0.5, -0.5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(EDGE); s.set_linewidth(0.5)
            fig.text((x + i * (w + gap)) / FIG_WIDTH, (y + h + 0.03) / height, title, fontsize=7.5, color=INK, ha="left", va="bottom")
            if i == 0:
                scale_bar(ax, crop.shape, m_per_px, spec["bar"], corner)
        fig.text((x + 1.5 * w + gap) / FIG_WIDTH, (y - 0.04) / height, spec["label"], fontsize=7.5, color=INK, ha="center", va="top")
        return y

    y_side = block(side, 0.0, sw, sh, side_scale, SIDE, "ll")
    block(top, x_top, tw, th, top_scale, TOP, "lr")
    handles = [Patch(facecolor=PALETTE[c] / 255, edgecolor="none", label=NAMES.get(c, CLASSES[c])) for c in LEGEND_CLASSES]
    handles += [Patch(facecolor=np.array(OK_RGB) / 255, edgecolor="none", label="correct (error panel)"),
                Patch(facecolor=np.array(ERR_RGB) / 255, edgecolor="none", label="wrong (error panel)")]
    left_width = 3 * sw + 2 * gap
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.5, frameon=False, handlelength=1.1, handleheight=0.9,
               columnspacing=1.2, labelspacing=0.35, borderaxespad=0.0,
               bbox_to_anchor=(0.5 * left_width / FIG_WIDTH, (y_side - 0.30) / height))
    written = []
    for ext in formats:
        fig.savefig(f"{out_base}.{ext}", dpi=dpi, facecolor="white")
        written.append(f"{out_base}.{ext}")
    plt.close(fig)
    return {"files": written, "height_in": height, "side_panel_in": (sw, sh), "top_panel_in": (tw, th),
            "side_width_m": side[0].shape[1] * side_scale, "top_size_m": (top[0].shape[1] * top_scale, top[0].shape[0] * top_scale)}


def main():
    fig_dir = os.path.join(ROOT, "figures")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", default=os.path.join(fig_dir, SIDE["file"]))
    ap.add_argument("--top", default=os.path.join(fig_dir, TOP["file"]))
    ap.add_argument("--out", default=os.path.join(fig_dir, "qualitative_reference"))
    a = ap.parse_args()
    info = compose(a.side, a.top, a.out)
    for f in info["files"]:
        print("wrote", f)
    print(f"figure {FIG_WIDTH:g} x {info['height_in']:.2f} in; side view {info['side_width_m']:.1f} m wide; "
          f"top-down view {info['top_size_m'][0]:.0f} x {info['top_size_m'][1]:.0f} m")


if __name__ == "__main__":
    main()
