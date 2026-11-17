import importlib.util
import os

import numpy as np

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("render_predictions", os.path.join(HERE, "..", "scripts", "render_predictions.py"))
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def test_raster_keeps_the_deepest_point_per_cell():
    # two points in the same cell: the one with the larger depth wins; cells are 1 m
    u = np.array([0.2, 0.7, 2.5]); v = np.array([0.1, 0.9, 0.3]); depth = np.array([1.0, 5.0, 2.0])
    rows, cols, sel, H, W = rp.raster(u, v, depth, cell=1.0)
    assert (H, W) == (1, 3)
    assert set(sel.tolist()) == {1, 2}            # point 0 is hidden behind point 1
    assert dict(zip(cols.tolist(), sel.tolist())) == {0: 1, 2: 2}


def test_images_mark_errors_and_ignore_unlabeled():
    rows = np.array([0, 0, 1]); cols = np.array([0, 1, 1])
    gt = np.array([1, 4, 255], np.uint8); pred = np.array([1, 6, 6], np.uint8)
    g, p, e, acc = rp.images(rows, cols, 2, 2, gt, pred)
    assert g.shape == (2, 2, 3) and acc == 0.5    # one correct, one wrong, one unlabeled
    assert tuple(g[1, 0]) == tuple(rp.PALETTE[1]) # origin flipped: row 0 is drawn at the bottom
    assert tuple(e[1, 1]) == rp.ERR_RGB           # the wrong point is red
    assert tuple(e[1, 0]) == rp.ERR_RGB           # its 4-neighbour is red by the 1-cell dilation
    assert tuple(e[0, 0]) == (255, 255, 255)      # empty cell (diagonal) stays white


def test_label_transfer_figure_from_the_repo_csv(tmp_path):
    """Figure of the motivating experiment: built from results/perturb_t5b5.csv, with the numbers the paper quotes."""
    s = importlib.util.spec_from_file_location("plot_label_transfer", os.path.join(HERE, "..", "scripts", "plot_label_transfer.py"))
    lt = importlib.util.module_from_spec(s)
    s.loader.exec_module(lt)
    data, noise = lt.load(os.path.join(HERE, "..", "results", "perturb_t5b5.csv"))
    assert noise == "fixed"                                                                  # the single-image file has no `noise` column
    assert [lv for lv, _ in data["translation_cm"]] == [0.0, 2.0, 5.0, 10.0, 25.0, 50.0]      # the unperturbed row comes first
    assert [lv for lv, _ in data["rotation_deg"]] == [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
    t10, r01 = dict(data["translation_cm"])[10.0], dict(data["rotation_deg"])[0.1]
    assert round(data["translation_cm"][0][1]["all"], 1) == 94.5 and round(data["translation_cm"][0][1]["structural"]) == 97
    assert round(t10["all"], 1) == 93.5 and round(t10["structural"]) == 47 and round(r01["all"], 1) == 93.9 and round(r01["structural"]) == 84
    pdf, png = lt.draw(data, str(tmp_path / "fig"))
    assert open(pdf, "rb").read(5) == b"%PDF-" and os.path.getsize(png) > 10_000
    # the zone-level file (scripts/label_transfer_zone.py): error model in a column, quartiles over the images as a band
    from gridnet import label_transfer as zone
    rec = lambda image, kind, level, share: {"image": image, "kind": kind, "level": level, "draw": 0, "hits": [int(1000 * share)] * 11,
                                              "total": [1000] * 11, "hits_all": int(11000 * share), "n_all": 11000}
    records = [rec(i, "none", 0.0, 0.95) for i in "abc"]
    for kind, levels in (("translation_cm", zone.TRANS_CM), ("rotation_deg", zone.ROT_DEG)):
        records += [rec(i, kind, lv, 0.9 - 0.1 * j - 0.05 * k) for j, lv in enumerate(levels) for k, i in enumerate("abc")]
    path = zone.write_csv(str(tmp_path / "zone.csv"), zone.summarize(records), "protocol")
    data2, noise2 = lt.load(path)
    assert noise2 == "protocol" and [lv for lv, _ in data2["rotation_deg"]] == [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
    first = data2["translation_cm"][1][1]
    assert abs(first["structural"] - 85.0) < 1e-9 and abs(first["structural_q25"] - 82.5) < 1e-9 and abs(first["structural_q75"] - 87.5) < 1e-9
    pdf2, png2 = lt.draw(data2, str(tmp_path / "fig2"), noise2)
    assert open(pdf2, "rb").read(5) == b"%PDF-" and os.path.getsize(png2) > 10_000
    # the run over all 852 images of the validation zones (label_transfer_v1): the figure and the numbers of Sec. 6.1
    data3, noise3 = lt.load(os.path.join(HERE, "..", "results", "label_transfer_v1", "label_transfer_val.csv"))
    assert noise3 == "protocol" and [lv for lv, _ in data3["translation_cm"]] == [0.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    clean, t10, r01 = data3["translation_cm"][0][1], dict(data3["translation_cm"])[10.0], dict(data3["rotation_deg"])[0.1]
    assert [round(clean[c], 1) for c in ("all", "conductor", "structural", "insulator")] == [95.6, 92.0, 91.7, 84.6]
    assert [round(t10[c], 1) for c in ("all", "conductor", "structural", "insulator")] == [94.8, 70.4, 55.8, 75.3]
    assert [round(r01[c], 1) for c in ("all", "conductor", "structural", "insulator")] == [94.7, 70.6, 65.3, 76.4]
    assert round(t10["structural_q25"], 1) == 19.3 and round(t10["structural_q75"], 1) == 82.0
    pdf3, png3 = lt.draw(data3, str(tmp_path / "fig3"), noise3)
    assert open(pdf3, "rb").read(5) == b"%PDF-" and os.path.getsize(png3) > 10_000


def test_protocol_figures_of_the_paper(tmp_path):
    """Benchmark and fix figures: one line per head type = mean over its seeds, band = range between the seeds."""
    import json
    s = importlib.util.spec_from_file_location("plot_paper_protocol", os.path.join(HERE, "..", "scripts", "plot_paper_protocol.py"))
    pp = importlib.util.module_from_spec(s)
    s.loader.exec_module(pp)
    val = {("a", "trans", 0.0, "mean_all"): 80.0, ("a", "trans", 0.5, "mean_all"): 70.0,
           ("b", "trans", 0.0, "mean_all"): 84.0, ("b", "trans", 0.5, "mean_all"): 76.0}
    assert pp.group_curve(val, ["a", "b", "missing"], "trans", "mean_all", [0.0, 0.5]) == ([82.0, 73.0], [80.0, 70.0], [84.0, 76.0])
    assert pp.group_curve(val, ["missing"], "trans", "mean_all", [0.0, 0.5]) is None
    summary = json.load(open(os.path.join(HERE, "..", "results", f"protocol_{pp.mt.EXACT_RUN}_summary.json")))
    labels = pp.draw(summary, "benchmark", str(tmp_path / "bench"))
    assert labels == ["LiDAR-only", "ImageVote", "Late fusion", "Feature concatenation", "Window 3×3", "Window 5×5"]
    labels = pp.draw(summary, "fix", str(tmp_path / "fix"))
    assert len(labels) == 9 and labels[0] == "LiDAR-only" and labels[1:3] == ["Late fusion, delivered poses", "Late fusion, pose noise"]
    for name in ("bench", "fix"):
        assert open(tmp_path / f"{name}.pdf", "rb").read(5) == b"%PDF-" and os.path.getsize(tmp_path / f"{name}.png") > 20_000
    # update 51: the figures are 3.7 in high (4.5 before); the axis labels of the two rows take two lines so that they
    # do not run into each other at that height, and the room above the panels follows the rows of the legend
    assert pp.FIG_SIZE == (7.2, 3.7) and [label.count("\n") for _, label in pp.ROWS] == [1, 1]
    assert [cls for cls, _ in pp.ROWS] == ["mean_electrical", "mean_all"]


def test_reference_figure_is_cut_from_two_renderings(tmp_path):
    """Fig. 4 of the paper (scripts/compose_reference_figure.py): the three panels of a side view and of a top-down view,
    cut out of the renderings of panel_figure by their frames, with the scale read from the tick marks."""
    s = importlib.util.spec_from_file_location("compose_reference_figure", os.path.join(HERE, "..", "scripts", "compose_reference_figure.py"))
    cr = importlib.util.module_from_spec(s)
    s.loader.exec_module(cr)
    assert cr._runs(np.array([3, 4, 5, 9, 12, 13])) == [(3, 5), (9, 9), (12, 13)] and cr._runs(np.array([], int)) == []
    rng = np.random.default_rng(0)

    def rendering(h, w, extent, name):
        imgs = [rp.PALETTE[rng.integers(0, 11, (h, w))].astype(np.uint8) for _ in range(3)]
        rp.panel_figure(imgs, ["Ground truth", "Prediction", "Errors"], extent, str(tmp_path / name), "a title", [0, 1, 6])
        return imgs

    # the shapes and extents of the two renderings of the paper: 70 m across with a tick every 10 m, 344 m with one every 50 m
    side = rendering(53, 70, (-35.0, 35.0, 25.0, 78.0), "side.png")
    top = rendering(158, 69, (0.0, 344.0, 0.0, 788.0), "top.png")
    for name, imgs, step, width_m in (("side.png", side, cr.SIDE["tick_step"], 70.0), ("top.png", top, cr.TOP["tick_step"], 344.0)):
        rgb = cr.load_rgb(str(tmp_path / name))
        fr = cr.frames(rgb)
        assert len(fr) == 3 and all(a[1] <= b[0] for a, b in zip(fr, fr[1:]))                 # three frames, left to right
        crops, m_per_px = cr.panels(str(tmp_path / name), step)
        assert len({c.shape for c in crops}) == 1
        assert abs(crops[0].shape[1] * m_per_px - width_m) < 0.01 * width_m                   # the scale of the tick marks
        for crop, img in zip(crops, imgs):                                                    # the pixels of the panel, unchanged
            assert tuple(crop[2, 2]) == tuple(img[0, 0]) and tuple(crop[-3, -3]) == tuple(img[-1, -1])
    info = cr.compose(str(tmp_path / "side.png"), str(tmp_path / "top.png"), str(tmp_path / "ref"))
    assert open(tmp_path / "ref.pdf", "rb").read(5) == b"%PDF-" and os.path.getsize(tmp_path / "ref.png") > 20_000
    assert abs(info["side_width_m"] - 70.0) < 0.7 and abs(info["top_size_m"][0] - 344.0) < 3.5
    assert info["height_in"] < 2.2 and info["side_panel_in"][0] > info["top_panel_in"][0]
    # a picture without frames is refused, not cut somewhere
    import matplotlib.image as mpimg
    mpimg.imsave(str(tmp_path / "blank.png"), np.full((50, 80, 3), 255, np.uint8))
    try:
        cr.frames(cr.load_rgb(str(tmp_path / "blank.png")))
        raise AssertionError("no error for a picture without frames")
    except ValueError as e:
        assert "frame lines" in str(e)
