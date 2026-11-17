"""Qualitative figure of the paper: the predictions around one pylon, with the delivered poses and under pose error.

  python scripts/plot_qualitative.py --list DIR                    # every crop in DIR: size, share of power-line points wrong
  python scripts/plot_qualitative.py --crop DIR/t1z8_crop0.npz     # -> figures/qualitative.{pdf,png}
      [--models concat_n,concat_a] [--levels trans:0,trans:0.5,rot:1] [--style errors|classes]
      [--azimuth 90] [--elevation 0] [--half-width 0] [--slab 0] [--cell 0.10] [--thick 1] [--thick-wrong 2] [--out BASE]
  python scripts/plot_qualitative.py --paper                       # the figure of the paper from results/qualitative/
  python scripts/plot_qualitative.py --crop A.npz B.npz --panels concat_n:trans:0,concat_n:rot:1,concat_a:rot:1
      [--half-width 22,35] [--row-labels "zone t1z8,zone t4z1"]     # one row per crop, one column per panel

Input: the crop files of `scripts/run_protocol.py --dump-pred` (gridnet/qualitative.py; notebook colab_qualitative): the
predictions of draw 0 of the protocol for the subset points around a pylon. numpy + matplotlib, no torch.

Layout. Several levels: one row per model (its name over the row), one column per level, the ground truth in the first
column. One level: one row, one column per model. With `--panels`: one row per crop (its ground truth first), one
column per (model, sweep, level), for several pylons in one figure. Every panel is an orthographic view (default: across the line, so
the conductors run left to right), drawn as a z-buffer raster of `--cell` metres: the surroundings first, the
power-line points (ground truth pylon, conductor, structural cable, insulator) over them and grown by `--thick` cells
so that cables stay visible in print. Style "errors" (default): a power-line point is grey when its predicted class is
right and red when it is not; a point of the surroundings that is predicted as a power-line class is blue; the other
surroundings are light grey whatever their prediction. Wrong points are drawn last (grown by `--thick-wrong`), so a
wrong point is never hidden by a right one in front of it - the picture overstates the wrong AREA; the percentage
printed in each panel is the exact share for the crop (every subset point within the radius, with its sampling
weight), not a count of pixels. Style "classes": every point in the colour of its predicted class.
The surroundings are a sparse sample of the scene (the protocol subset), the power-line structures are dense.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from gridnet import qualitative as Q  # noqa: E402
from gridnet.classes import CLASSES, PALETTE  # noqa: E402

INK, INK2, EDGE = "#0b0b0b", "#52514e", "#c9c8c3"
RIGHT_RGB, WRONG_RGB, ALARM_RGB, BACK_RGB = (143, 142, 137), (214, 32, 32), (30, 100, 200), (224, 223, 218)
REF_LABELS = {"lidar": "LiDAR-only", "imagevote": "ImageVote"}
# the figure of the paper (--paper): the crops in results/qualitative/, chosen from the twelve pylons of the run qual_v1
# by the share of power-line points that the first model below gets wrong at the largest rotation error (crops.txt
# there lists all twelve): the pylon with the largest share and one of the two middle ones
PAPER = {"crops": ["t1z8_crop0.npz", "t4z1_crop0.npz"], "row_labels": ["zone t1z8", "zone t4z1"],
         "panels": [("concat_n", "trans", 0.0), ("concat_n", "rot", 1.0), ("concat_a", "rot", 1.0)], "out": "qualitative_pose_error"}


def _head_groups():
    """[(head type label, block heading, names)] of the paper tables; [] when scripts/make_tables.py cannot be loaded."""
    try:
        spec = importlib.util.spec_from_file_location("plot_paper_protocol", os.path.join(ROOT, "scripts", "plot_paper_protocol.py"))
        pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
    except Exception:                                    # the figure does not depend on the tables
        return []
    legend = {h[0]: h[1] for h in pp.HEADS}
    out, block = [], ""
    for label, names in pp.mt.FIX_GROUPS:
        if names is None:
            block = label
        else:
            out.append((legend.get(label, label), block, names))
    return out


def model_label(name, groups=None):
    """Two-line label of a model: head type / how it was trained (the name itself for a model outside the tables)."""
    if name in REF_LABELS:
        return REF_LABELS[name]
    for label, block, names in (_head_groups() if groups is None else groups):
        if name in names:
            return f"{label},\n{'pose-noise augmentation' if block.startswith('Pose-noise') else 'delivered poses in training'}"
    return name


def level_label(sweep, level):
    if float(level) == 0.0:
        return "Delivered poses"
    return {"trans": f"Translation error $\\sigma_t$ = {100 * level:g} cm", "rot": f"Rotation error $\\sigma_r$ = {level:g}°",
            "drift": f"Drift, {100 * level:g} cm at the end of the line"}.get(sweep, f"{sweep} {level:g}")


def crop_key(crop, model, sweep, level):
    """The key of (model, sweep, level) in the crop; the delivered poses are the level 0 of any sweep."""
    k = Q.key(model, sweep, level)
    if k in crop["preds"]:
        return k
    if float(level) == 0.0:
        for other in sorted(crop["preds"]):
            m, _, lv = Q.split_key(other)
            if m == model and lv == 0.0:
                return other
    return None


# ----------------------------------------------------------------------------- raster
def zbuffer(col, row, depth, width):
    """Per occupied cell the point nearest to the viewer (largest depth): (flat cell index, point index)."""
    if len(col) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    cell = row.astype(np.int64) * width + col
    order = np.lexsort((depth, cell))
    c = cell[order]
    last = np.empty(len(c), bool)
    last[:-1] = c[1:] != c[:-1]
    last[-1] = True
    return c[last], order[last]


def grow(index_image, r):
    """Empty cells (-1) within r cells of an occupied one take the value of that neighbour (the nearest first)."""
    out = index_image.copy()
    if r <= 0:
        return out
    H, W = out.shape
    offsets = sorted(((dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if (dy or dx) and dy * dy + dx * dx <= r * r + r),
                     key=lambda o: o[0] ** 2 + o[1] ** 2)
    for dy, dx in offsets:
        src = index_image[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        dst = out[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
        m = (dst < 0) & (src >= 0)
        dst[m] = src[m]
    return out


class Scene:
    """One view of a crop: the raster cell of every kept point and, per layer, the point every cell shows."""

    def __init__(self, crop, azimuth=90.0, elevation=0.0, cell=0.10, half_width=0.0, slab=0.0):
        xyz, lab = crop["xyz"], crop["labels"]
        right, up, towards = Q.view_axes(Q.line_direction(xyz, lab), azimuth, elevation)
        p = xyz.astype(np.float64)
        u, v, depth = p @ right, p @ up, p @ towards
        keep = np.ones(len(u), bool)
        if half_width > 0:
            keep &= np.abs(u) <= half_width
        if slab > 0:                                     # distance from the vertical plane through the pylon that faces the viewer
            h = towards[:2] / max(np.linalg.norm(towards[:2]), 1e-9)
            keep &= np.abs(p[:, :2] @ h) <= slab
        assert keep.any(), "no point left in the view"
        self.idx = np.flatnonzero(keep)
        u, v, self.depth = u[keep], v[keep], depth[keep]
        self.cell = float(cell)
        u0, v0 = u.min(), v.min()
        self.col = np.floor((u - u0) / cell).astype(np.int64)
        self.row = np.floor((v - v0) / cell).astype(np.int64)
        self.W, self.H = int(self.col.max()) + 1, int(self.row.max()) + 1
        self.extent = (u0, u0 + self.W * cell, v0, v0 + self.H * cell)
        self.labels = lab[self.idx]
        self.electrical = np.isin(self.labels, Q.ELECTRICAL)
        self.back = self._layer(~self.electrical)
        self.top = self._layer(self.electrical)

    def _layer(self, mask):
        sel = np.flatnonzero(mask)
        cells, pts = zbuffer(self.col[sel], self.row[sel], self.depth[sel], self.W)
        return cells, sel[pts]

    def image(self, rgb, emphasis=(), thick=1, thick_wrong=2):
        """(H, W, 3) uint8 image with the top row first. `rgb` (n, 3): the colour of every kept point. `emphasis`:
        boolean masks over the kept points, drawn last and in this order, grown by `thick_wrong` cells."""
        n = self.H * self.W
        img = np.full((n, 3), 255, np.uint8)
        cells, pts = self.back
        img[cells] = rgb[pts]
        layers = [(self.top, thick)] + [(self._layer(m), thick_wrong) for m in emphasis if m is not None and m.any()]
        for (cells, pts), r in layers:
            index = np.full(n, -1, np.int64)
            index[cells] = pts
            index = grow(index.reshape(self.H, self.W), r).ravel()
            m = index >= 0
            img[m] = rgb[index[m]]
        return img.reshape(self.H, self.W, 3)[::-1]


def panel_colours(scene, crop, key=None, style="errors"):
    """(rgb per kept point, masks to draw last) of a panel; key None = the ground truth in class colours."""
    gt = scene.labels
    if key is None:
        return PALETTE[gt], ()
    pred = crop["preds"][key][scene.idx]
    wrong = scene.electrical & (pred != gt)
    alarm = ~scene.electrical & np.isin(pred, Q.ELECTRICAL)
    if style == "classes":
        return PALETTE[pred], (alarm, wrong)
    rgb = np.where(scene.electrical[:, None], np.array(RIGHT_RGB, np.uint8), np.array(BACK_RGB, np.uint8))
    rgb[alarm] = ALARM_RGB
    rgb[wrong] = WRONG_RGB
    return rgb, (alarm, wrong)


# ----------------------------------------------------------------------------- figure
def legend_handles(class_ids, style):
    """Legend entries: the classes of the ground-truth panels, then the colours of the prediction panels."""
    handles = [Patch(facecolor=PALETTE[c] / 255, edgecolor="none", label=CLASSES[c]) for c in sorted(class_ids)]
    if style == "errors":
        handles += [Patch(facecolor=np.array(RIGHT_RGB) / 255, edgecolor="none", label="power-line point, class right"),
                    Patch(facecolor=np.array(WRONG_RGB) / 255, edgecolor="none", label="power-line point, class wrong"),
                    Patch(facecolor=np.array(ALARM_RGB) / 255, edgecolor="none", label="surroundings predicted as power line"),
                    Patch(facecolor=np.array(BACK_RGB) / 255, edgecolor="none", label="other surroundings")]
    return handles


def scale_bar(ax, scene):
    """A 10 m bar (5 m in a view narrower than 30 m) in the lower left corner of a panel."""
    x0, x1, y0, y1 = scene.extent
    bar = 10.0 if x1 - x0 >= 30 else 5.0
    bx, by = x0 + 0.04 * (x1 - x0), y0 + 0.07 * (y1 - y0)
    ax.plot([bx, bx + bar], [by, by], color=INK, lw=1.1, solid_capstyle="butt")
    ax.text(bx + bar + 0.015 * (x1 - x0), by, f"{bar:g} m", ha="left", va="center", fontsize=6, color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.8})
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def note(ax, text, x, y, ha, va):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6.5, color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2})


def panel_axes(fig, x, y, w, h):
    """Axes at (x, y) with size (w, h) in inches, without ticks."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(EDGE); sp.set_linewidth(0.5)
    return ax


def _per_crop(value, n):
    """A view parameter for n crops: one number for all of them, or one per crop."""
    values = list(value) if isinstance(value, (list, tuple)) else [value] * n
    assert len(values) == n, f"{len(values)} values for {n} crops"
    return [float(v) for v in values]


def figure_rows(crops, panels, out_base, style="errors", azimuth=90.0, elevation=0.0, cell=0.10, half_width=0.0, slab=0.0,
                thick=1, thick_wrong=2, width=7.2, dpi=300, groups=None, row_labels=None):
    """Several pylons in one figure: one row per crop - its ground truth, then one panel per (model, sweep, level) of
    `panels`. `azimuth`, `elevation`, `half_width` and `slab` are one number for every crop or a list with one per
    crop. Writes <out_base>.pdf / .png; returns {"panels": [(row, model, sweep, level, share wrong)], "rows", "cols"}.
    Raises when a crop lacks one of the panels: a figure with a hole would not be noticed in a paper."""
    n = len(crops)
    az, el, hw, sl = (_per_crop(v, n) for v in (azimuth, elevation, half_width, slab))
    scenes = [Scene(c, azimuth=az[i], elevation=el[i], cell=cell, half_width=hw[i], slab=sl[i]) for i, c in enumerate(crops)]
    keys = [[crop_key(c, m, s, lv) for m, s, lv in panels] for c in crops]
    missing = [f"{c['zone']}: {m}|{s}|{lv:g}" for c, row in zip(crops, keys) for (m, s, lv), k in zip(panels, row) if k is None]
    if missing:
        raise ValueError(f"not in the crops: {missing}")
    row_labels = list(row_labels) if row_labels else [f"zone {c['zone']}" for c in crops]
    n_cols = 1 + len(panels)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    left, right, gutter, gap, hgap = 0.04, 0.04, 0.10, 0.06, 0.10          # inches
    heads = [model_label(m, groups) for m, _, _ in panels]
    lines = max(t.count("\n") + 1 for t in heads)
    top = 0.05 + 0.125 * (lines + 1)                                       # the model, then the pose error in its own line
    pw = (width - left - right - gutter - gap * (n_cols - 2)) / n_cols
    ph = [pw * sc.H / sc.W for sc in scenes]
    dpi = max(int(dpi), int(np.ceil(max(sc.W for sc in scenes) / pw)))
    handles = legend_handles({int(c) for sc in scenes for c in set(sc.labels.tolist())}, style)
    legend_cols = 4
    bottom = 0.15 * int(np.ceil(len(handles) / legend_cols)) + 0.12
    height = top + sum(ph) + (n - 1) * hgap + bottom
    fig = plt.figure(figsize=(width, height), dpi=dpi)

    def col_x(c):
        return left + (0 if c == 0 else pw + gutter + (c - 1) * (pw + gap))

    fig.text(col_x(0) / width, (height - top + 0.035) / height, "Ground truth", fontsize=7.5, color=INK, ha="left", va="bottom")
    for c, (head, (m, s, lv)) in enumerate(zip(heads, panels), start=1):
        fig.text(col_x(c) / width, (height - 0.04) / height, head, fontsize=7.5, color=INK, ha="left", va="top", linespacing=1.15)
        fig.text(col_x(c) / width, (height - top + 0.035) / height, level_label(s, lv), fontsize=7, color=INK2, ha="left", va="bottom")
    out, y = [], height - top
    for r, (crop, scene) in enumerate(zip(crops, scenes)):
        y -= ph[r]
        ax = panel_axes(fig, col_x(0), y, pw, ph[r])
        rgb, _ = panel_colours(scene, crop)
        ax.imshow(scene.image(rgb, thick=thick), extent=scene.extent, interpolation="none", aspect="auto")
        scale_bar(ax, scene)
        note(ax, f"({chr(97 + r)}) {row_labels[r]}", 0.975, 0.06, "right", "bottom")
        for c, ((m, s, lv), k) in enumerate(zip(panels, keys[r]), start=1):
            ax = panel_axes(fig, col_x(c), y, pw, ph[r])
            rgb, emphasis = panel_colours(scene, crop, k, style)
            ax.imshow(scene.image(rgb, emphasis, thick=thick, thick_wrong=thick_wrong), extent=scene.extent, interpolation="none", aspect="auto")
            share = Q.wrong_share(crop["cm"][k])
            note(ax, f"{100 * share:.1f} % wrong", 0.975, 0.06, "right", "bottom")
            out.append((r, m, s, float(lv), share))
        y -= hgap
    fig.legend(handles=handles, loc="lower center", ncol=legend_cols, fontsize=6.2, frameon=False, handlelength=1.1, handleheight=0.9,
               columnspacing=1.4, labelspacing=0.3, bbox_to_anchor=(0.5, 0.0))
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)
    return {"panels": out, "rows": n, "cols": n_cols, "raster": [(sc.H, sc.W) for sc in scenes], "dpi": dpi}


def figure(crop, models, levels, out_base, style="errors", azimuth=90.0, elevation=0.0, cell=0.10, half_width=0.0, slab=0.0,
           thick=1, thick_wrong=2, width=7.2, dpi=300, groups=None):
    """Writes <out_base>.pdf and <out_base>.png. Returns {"panels": [(model, sweep, level, share wrong)], "rows", "cols",
    "skipped": [...]}; raises when the crop has none of the wanted panels."""
    scene = Scene(crop, azimuth=azimuth, elevation=elevation, cell=cell, half_width=half_width, slab=slab)
    wanted = [(m, s, lv, crop_key(crop, m, s, lv)) for m in models for s, lv in levels]
    skipped = [f"{m}|{s}|{lv:g}" for m, s, lv, k in wanted if k is None]
    have_models = [m for m in models if any(k for mm, _, _, k in wanted if mm == m)]
    have_levels = [(s, lv) for s, lv in levels if any(k for _, ss, ll, k in wanted if (ss, ll) == (s, lv))]
    if not have_models:
        raise ValueError(f"the crop holds none of {models} at {levels}; it has {sorted(crop['preds'])}")
    one_level = len(have_levels) == 1
    if one_level:                                        # one row: ground truth, then one column per model
        rows = [[(m, *have_levels[0]) for m in have_models]]
        col_titles = [model_label(m, groups) for m in have_models]
        row_titles = [level_label(*have_levels[0])]
    else:                                                # one row per model, one column per level
        rows = [[(m, s, lv) for s, lv in have_levels] for m in have_models]
        col_titles = [level_label(s, lv) for s, lv in have_levels]
        row_titles = [model_label(m, groups) for m in have_models]
    n_rows, n_cols = len(rows), 1 + len(rows[0])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    # inches. From the top: column heads, then per row a heading line and its panels; the legend under everything
    left, right, gutter, gap, hgap, head = 0.04, 0.04, 0.10, 0.06, 0.07, 0.15
    top = 0.05 + 0.125 * max(t.count("\n") + 1 for t in col_titles)
    pw = (width - left - right - gutter - gap * (n_cols - 2)) / n_cols
    ph = pw * scene.H / scene.W
    dpi = max(int(dpi), int(np.ceil(scene.W / pw)))      # a raster cell is at least one pixel: thin structures are not resampled away
    handles = legend_handles(set(scene.labels.tolist()), style)
    legend_cols = 4
    bottom = 0.15 * int(np.ceil(len(handles) / legend_cols)) + 0.12
    height = top + n_rows * (head + ph) + (n_rows - 1) * hgap + bottom
    fig = plt.figure(figsize=(width, height), dpi=dpi)

    def col_x(c):
        return left + (0 if c == 0 else pw + gutter + (c - 1) * (pw + gap))

    def row_top(r):                                      # top of the heading line of row r
        return height - top - r * (head + ph + hgap)

    def axes_at(x, y):
        return panel_axes(fig, x, y, pw, ph)

    def show(ax, img):
        ax.imshow(img, extent=scene.extent, interpolation="none", aspect="auto")

    # ground truth: first column, centred over the rows
    gt_y = (row_top(0) - head + row_top(n_rows - 1) - head - ph) / 2 - ph / 2
    ax = axes_at(col_x(0), gt_y)
    rgb, _ = panel_colours(scene, crop)
    show(ax, scene.image(rgb, thick=thick))
    fig.text(col_x(0) / width, (gt_y + ph + 0.035) / height, "Ground truth", fontsize=7.5, color=INK, ha="left", va="bottom")
    scale_bar(ax, scene)

    for c, title in enumerate(col_titles, start=1):      # column heads
        fig.text(col_x(c) / width, (height - 0.04) / height, title, fontsize=7.5, color=INK, ha="left", va="top", linespacing=1.15)
    panels = []
    for r, row in enumerate(rows):
        fig.text(col_x(1) / width, (row_top(r) - 0.025) / height, row_titles[r].replace("\n", " "), fontsize=7, color=INK2, ha="left", va="top")
        for c, (m, s, lv) in enumerate(row, start=1):
            ax = axes_at(col_x(c), row_top(r) - head - ph)
            k = crop_key(crop, m, s, lv)
            if k is None:
                ax.text(0.5, 0.5, "not in the crop", transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=INK2)
                continue
            rgb, emphasis = panel_colours(scene, crop, k, style)
            show(ax, scene.image(rgb, emphasis, thick=thick, thick_wrong=thick_wrong))
            share = Q.wrong_share(crop["cm"][k])
            note(ax, f"{100 * share:.1f} % wrong", 0.975, 0.06, "right", "bottom")
            panels.append((m, s, float(lv), share))
    fig.legend(handles=handles, loc="lower center", ncol=legend_cols, fontsize=6.2, frameon=False, handlelength=1.1, handleheight=0.9,
               columnspacing=1.4, labelspacing=0.3, bbox_to_anchor=(0.5, 0.0))
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)
    return {"panels": panels, "rows": n_rows, "cols": n_cols, "skipped": skipped, "raster": (scene.H, scene.W), "dpi": dpi}


# ----------------------------------------------------------------------------- list
def crop_files(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz") and "_crop" in f and ".tmp" not in f)


def list_crops(folder, out=print, json_path=None):
    """One block per crop: its size and the share of power-line points wrong per model and level. Returns the rows
    [{"file", "zone", "model", "sweep", "level", "wrong"}]; `json_path` also gets them, with one entry per crop
    (zone, centre, points, weighted points per class) - the table a figure caption can cite for ALL crops of a run
    when only the drawn ones are kept."""
    rows, meta = [], []
    for path in crop_files(folder):
        crop = Q.load_crop(path)
        counts = crop["class_points"]
        meta.append({"file": os.path.basename(path), "zone": crop["zone"], "centre": [float(x) for x in crop["centre"]],
                     "radius": crop["radius"], "points_kept": int(len(crop["labels"])), "points_subset": crop["n_subset"],
                     "class_points": {CLASSES[c]: float(counts[c]) for c in range(Q.NUM) if counts[c] > 0}})
        keys = [Q.split_key(k) for k in sorted(crop["preds"])]
        order = {"trans": 0, "rot": 1, "drift": 2}                # the delivered poses first, then the sweeps in the order of the protocol
        levels = sorted({(s, lv) for _, s, lv in keys}, key=lambda t: (t[1] != 0.0, order.get(t[0], 9), t[0], t[1]))
        models = list(dict.fromkeys(m for m, _, _ in keys))
        out(f"{os.path.basename(path)}: zone {crop['zone']}, centre ({crop['centre'][0]:.0f}, {crop['centre'][1]:.0f}) m, radius {crop['radius']:g} m, "
            f"{len(crop['labels']):,} points kept of {crop['n_subset']:,} subset points, {os.path.getsize(path) / 2 ** 20:.1f} MB")
        out("   points per class (weighted): " + ", ".join(f"{CLASSES[c].lower()} {counts[c]:,.0f}" for c in range(Q.NUM) if counts[c] > 0))
        out(f"   power-line points wrong [%]  {'model':16s}" + "".join(f"{s + ' ' + format(lv, 'g'):>12s}" for s, lv in levels))
        for m in models:
            cells = []
            for s, lv in levels:
                k = Q.key(m, s, lv)
                share = Q.wrong_share(crop["cm"][k]) if k in crop["cm"] else float("nan")
                cells.append(share)
                rows.append({"file": os.path.basename(path), "zone": crop["zone"], "model": m, "sweep": s, "level": lv, "wrong": share})
            out(f"   {'':29s}{m:16s}" + "".join(f"{100 * v:12.1f}" for v in cells))
    if not rows:
        out(f"no crop files in {folder}")
    if json_path:
        with open(json_path, "w") as fh:
            json.dump({"crops": meta, "rows": rows}, fh, indent=1)
    return rows


def parse_panels(text):
    """"concat_n:trans:0,concat_a:rot:1" -> [("concat_n", "trans", 0.0), ("concat_a", "rot", 1.0)]"""
    out = []
    for item in text.split(","):
        if item.strip():
            model, sweep, level = item.strip().rsplit(":", 2)
            out.append((model, sweep, float(level)))
    return out


def crop_statistics(table, model="concat_n", sweep="rot", level=1.0):
    """[(file, share of power-line points that `model` gets wrong at the level)] sorted by the share, from the table
    of a run (the "rows" of `list_crops`, or the path of its JSON file)."""
    rows = json.load(open(table))["rows"] if isinstance(table, str) else table
    sel = [(r["file"], r["wrong"]) for r in rows if (r["model"], r["sweep"]) == (model, sweep) and abs(r["level"] - level) < 1e-9]
    return sorted(sel, key=lambda r: r[1])


def paper_figure(folder=None, out_dir=None, dpi=300):
    """The qualitative figure of the paper from the crops in results/qualitative/; returns the info of figure_rows."""
    folder = folder or os.path.join(ROOT, "results", "qualitative")
    crops = [Q.load_crop(os.path.join(folder, f)) for f in PAPER["crops"]]
    out = os.path.join(out_dir or os.path.join(ROOT, "figures"), PAPER["out"])
    info = figure_rows(crops, PAPER["panels"], out, row_labels=PAPER["row_labels"], dpi=dpi)
    info["out"] = out
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default=None, metavar="DIR", help="print a table of every crop in DIR and stop")
    ap.add_argument("--json", default=None, help="with --list: also write the table as JSON to this file")
    ap.add_argument("--paper", action="store_true", help="draw the figure of the paper (figures/qualitative_pose_error) from results/qualitative/")
    ap.add_argument("--crop", default=None, nargs="+", help="crop file(s) (<zone>_crop<i>.npz); several only with --panels")
    ap.add_argument("--panels", default=None, help="model:sweep:level,... -> one row per crop, one column per panel")
    ap.add_argument("--row-labels", default=None, help="with --panels: comma-separated labels of the rows (default: zone <name>)")
    ap.add_argument("--models", default="concat_n,concat_a", help="models, in the order of the rows (one level: of the columns)")
    ap.add_argument("--levels", default="trans:0,trans:0.5,rot:1", help="sweep:level pairs, in the order of the columns")
    ap.add_argument("--style", choices=["errors", "classes"], default="errors")
    ap.add_argument("--azimuth", default="90", help="viewer's angle from the line direction: 90 = across the line, 0 = along it (with --panels: one per crop, comma-separated)")
    ap.add_argument("--elevation", default="0", help="viewer's angle above the horizon")
    ap.add_argument("--half-width", default="0", help="keep |horizontal view coordinate| <= this [m]; 0 = the whole crop")
    ap.add_argument("--slab", default="0", help="keep the points within this distance of the view plane through the pylon [m]; 0 = all")
    ap.add_argument("--cell", type=float, default=0.10, help="raster cell [m]")
    ap.add_argument("--thick", type=int, default=1, help="power-line points are grown by this many cells")
    ap.add_argument("--thick-wrong", type=int, default=2, help="wrong points are grown by this many cells")
    ap.add_argument("--width", type=float, default=7.2, help="figure width [in]")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "qualitative"), help="output base name (.pdf and .png are added)")
    a = ap.parse_args()
    if a.list:
        list_crops(a.list, json_path=a.json)
        return
    if a.paper:
        info = paper_figure()
        print(f"wrote {info['out']}.pdf / .png  ({info['rows']} pylons x {info['cols']} panels)")
        for r, m, s, lv, share in info["panels"]:
            print(f"   {PAPER['row_labels'][r]:10s} {m:10s} {s:5s} {lv:5g}   {100 * share:5.1f} % of the power-line points wrong")
        return
    assert a.crop, "give --crop FILE, --paper or --list DIR"

    def view(text, n):
        v = [float(x) for x in str(text).split(",")]
        return v[0] if n == 1 else (v * n if len(v) == 1 else v)
    if a.panels:
        crops = [Q.load_crop(f) for f in a.crop]
        n = len(crops)
        info = figure_rows(crops, parse_panels(a.panels), a.out, style=a.style, azimuth=view(a.azimuth, n), elevation=view(a.elevation, n),
                           cell=a.cell, half_width=view(a.half_width, n), slab=view(a.slab, n), thick=a.thick, thick_wrong=a.thick_wrong,
                           width=a.width, row_labels=a.row_labels.split(",") if a.row_labels else None)
        print(f"wrote {a.out}.pdf / .png  ({info['rows']} crops x {info['cols']} panels)")
        for r, m, s, lv, share in info["panels"]:
            print(f"   {crops[r]['zone']:8s} {m:16s} {s:5s} {lv:5g}   {100 * share:5.1f} % of the power-line points wrong")
        return
    assert len(a.crop) == 1, "several crops need --panels"
    crop = Q.load_crop(a.crop[0])
    info = figure(crop, [m.strip() for m in a.models.split(",") if m.strip()], Q.parse_levels(a.levels), a.out, style=a.style,
                  azimuth=view(a.azimuth, 1), elevation=view(a.elevation, 1), cell=a.cell, half_width=view(a.half_width, 1),
                  slab=view(a.slab, 1), thick=a.thick, thick_wrong=a.thick_wrong, width=a.width)
    print(f"wrote {a.out}.pdf / .png  (zone {crop['zone']}, {info['rows']} x {info['cols']} panels, raster {info['raster'][1]} x {info['raster'][0]} cells)")
    for m, s, lv, share in info["panels"]:
        print(f"   {m:16s} {s:5s} {lv:5g}   {100 * share:5.1f} % of the power-line points wrong")
    if info["skipped"]:
        print("   not in the crop:", ", ".join(info["skipped"]))


if __name__ == "__main__":
    main()
