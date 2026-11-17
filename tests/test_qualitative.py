"""Crops for the qualitative figure (gridnet/qualitative.py) and their drawing (scripts/plot_qualitative.py): numpy only."""
import importlib.util
import os

import numpy as np

from gridnet import qualitative as Q

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("plot_qualitative", os.path.join(HERE, "..", "scripts", "plot_qualitative.py"))
pq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pq)


def synth_scene(seed=0, line_deg=30.0):
    """A power line through two pylons 120 m apart (the second one smaller), with ground and vegetation around it.
    Returns (xyz float32 zone-relative, labels uint8, weights, pylon centres (2, 2))."""
    rng = np.random.default_rng(seed)
    d = np.array([np.cos(np.radians(line_deg)), np.sin(np.radians(line_deg))])
    nrm = np.array([-d[1], d[0]])
    centres = np.array([200.0, 300.0]) + np.outer([0.0, 120.0], d)
    parts, labels = [], []

    def add(p, c):
        parts.append(np.asarray(p, np.float64)); labels.append(np.full(len(p), c, np.uint8))

    for c, n_py in zip(centres, (6000, 2500)):                       # pylon: four legs that lean inwards + two cross-arms
        t = rng.uniform(0, 1, n_py)
        leg = rng.integers(0, 4, n_py)
        sx, sy = np.where(leg % 2 == 0, -1.0, 1.0), np.where(leg < 2, -1.0, 1.0)
        base = 3.0 * (1 - 0.8 * t)
        xy = c + (sx * base)[:, None] * d + (sy * base)[:, None] * nrm
        add(np.column_stack([xy, 30.0 * t]) + rng.normal(0, 0.03, (n_py, 3)), 0)
        for h in (22.0, 28.0):
            s = rng.uniform(-8, 8, 600)
            add(np.column_stack([c + s[:, None] * nrm, np.full(600, h)]) + rng.normal(0, 0.03, (600, 3)), 0)
            for off in (-8.0, 8.0):                                  # insulators hang from the ends of the arms
                add(np.column_stack([np.tile(c + off * nrm, (60, 1)), h - rng.uniform(0, 1.5, 60)]), 3)
    s = rng.uniform(-60, 180, 9000)                                  # conductors: six cables with a sag between the pylons
    which = rng.integers(0, 6, 9000)
    off = np.array([-8.0, 8.0])[which % 2]
    h = np.array([20.5, 26.5, 20.5, 26.5, 20.5, 26.5])[which] - 4.0 * np.sin(np.pi * np.clip(s / 120.0, 0, 1)) ** 2
    add(np.column_stack([centres[0] + s[:, None] * d + off[:, None] * nrm, h]), 1)
    s = rng.uniform(-60, 180, 1500)                                  # one structural (guard) cable over the top
    add(np.column_stack([centres[0] + s[:, None] * d, np.full(1500, 30.5)]), 2)
    n_bg = 6000                                                      # sparse surroundings: herbaceous ground, low and high vegetation
    xy = centres[0] + rng.uniform(-80, 200, n_bg)[:, None] * d + rng.uniform(-60, 60, n_bg)[:, None] * nrm
    kind = rng.choice([6, 5, 4], n_bg, p=[0.6, 0.25, 0.15])
    add(np.column_stack([xy, np.where(kind == 6, 0.0, np.where(kind == 5, 1.0, 6.0)) * rng.uniform(0.2, 1.0, n_bg)]), 0)
    labels[-1][:] = kind
    xyz = np.concatenate(parts).astype(np.float32)
    lab = np.concatenate(labels)
    order = rng.permutation(len(lab))                                # a subset is in file order, not grouped by class
    w = np.where(np.isin(lab, Q.ELECTRICAL), 1.0, 40.0).astype(np.float32)
    return xyz[order], lab[order], w[order], centres


def fake_preds(xyz, lab, centres):
    """model "bad": the conductors and the structural cable within 20 m of the first pylon become low vegetation at the
    error levels; model "good": always right except for a few insulator points."""
    near = np.linalg.norm(xyz[:, :2] - centres[0], axis=1) < 20.0
    bad = lab.copy(); bad[near & np.isin(lab, (1, 2))] = 5
    good = lab.copy(); good[np.flatnonzero(lab == 3)[:10]] = 0
    return {"clean": {"bad": lab.copy(), "good": good}, "noisy": {"bad": bad, "good": good}}


def dump_scene(out_dir, **kw):
    xyz, lab, w, centres = synth_scene()
    preds = fake_preds(xyz, lab, centres)
    dump = Q.CropDump(out_dir, [("trans", 0.0), ("trans", 0.5), ("rot", 1.0)], **kw)
    dump(zone="zs", sweep="trans", level=0.0, draw=0, preds=preds["clean"], xyz=xyz, labels=lab, weights=w)
    dump(zone="zs", sweep="trans", level=0.5, draw=0, preds=preds["noisy"], xyz=xyz, labels=lab, weights=w)
    dump(zone="zs", sweep="trans", level=0.5, draw=1, preds=preds["clean"], xyz=xyz, labels=lab, weights=w)   # another draw: not kept
    dump(zone="zs", sweep="trans", level=0.25, draw=0, preds=preds["clean"], xyz=xyz, labels=lab, weights=w)  # level not wanted
    dump(zone="zs", sweep="rot", level=1.0, draw=0, preds=preds["noisy"], xyz=xyz, labels=lab, weights=w)
    return dump, xyz, lab, w, centres, preds


def test_pylon_centres_largest_first_and_fallback():
    xyz, lab, _, centres = synth_scene()
    found = Q.pylon_centres(xyz, lab, n=2)
    assert len(found) == 2
    assert np.linalg.norm(found[0] - centres[0]) < 2.0 and np.linalg.norm(found[1] - centres[1]) < 2.0     # the larger pylon first
    assert len(Q.pylon_centres(xyz, lab, n=1)) == 1
    assert len(Q.pylon_centres(xyz, lab, n=5)) == 2                                     # no third peak with enough points
    no_pylon = np.where(lab == 0, 4, lab)
    c = Q.pylon_centres(xyz, no_pylon, n=2)
    el = xyz[np.isin(no_pylon, Q.ELECTRICAL), :2]
    assert len(c) == 1 and np.allclose(c[0], np.median(el, axis=0))                     # fallback: middle of the power-line points
    c = Q.pylon_centres(xyz, np.full(len(lab), 6, np.uint8), n=2)
    assert len(c) == 1 and np.allclose(c[0], np.median(xyz[:, :2], axis=0))


def test_thin_indices_one_point_per_voxel_and_budget():
    rng = np.random.default_rng(0)
    rel = np.array([[0.01, 0.01, 0.01], [0.02, 0.03, 0.04], [0.26, 0.0, 0.0], [0.27, 0.01, 0.02], [1.0, 1.0, 1.0]])
    lab = np.array([1, 1, 6, 6, 0], np.uint8)
    assert Q.thin_indices(rel, lab, 0.05, 100, rng).tolist() == [0, 2, 4]              # the first point of every 5 cm voxel
    assert Q.thin_indices(rel, lab, 0.0, 100, rng).tolist() == [0, 1, 2, 3, 4]          # no thinning
    rel = rng.uniform(0, 50, (5000, 3)); lab = np.where(np.arange(5000) < 300, 1, 6).astype(np.uint8)
    keep = Q.thin_indices(rel, lab, 0.0, 1000, rng)
    assert len(keep) == 1000 and (lab[keep] == 1).sum() == 300 and np.all(np.diff(keep) > 0)   # power-line points stay
    lab = np.where(np.arange(5000) < 3000, 1, 6).astype(np.uint8)
    keep = Q.thin_indices(rel, lab, 0.0, 1000, rng)
    assert (lab[keep] == 1).sum() == 800 and (lab[keep] == 6).sum() == 200             # the surroundings keep a fifth of the budget
    lab = np.where(np.arange(5000) < 3000, 0, np.where(np.arange(5000) < 3300, 2, 6)).astype(np.uint8)
    keep = Q.thin_indices(rel, lab, 0.0, 1000, rng)                                    # a dense pylon gives way to the thin classes
    assert np.bincount(lab[keep], minlength=7)[[0, 2, 6]].tolist() == [500, 300, 200] and len(np.unique(keep)) == 1000


def test_crop_dump_files_hold_the_predictions_and_exact_statistics(tmp_path):
    dump, xyz, lab, w, centres, preds = dump_scene(str(tmp_path), radius=35.0, pylons=2, voxel=0.0, max_points=10**9)
    paths = dump.paths("zs")
    assert [os.path.basename(p) for p in paths] == ["zs_crop0.npz", "zs_crop1.npz"] and all(os.path.exists(p) for p in paths)
    assert not [f for f in os.listdir(tmp_path) if ".tmp" in f]
    crop = Q.load_crop(paths[0])
    assert crop["zone"] == "zs" and crop["radius"] == 35.0 and np.linalg.norm(crop["centre"] - centres[0]) < 2.0
    assert sorted(crop["preds"]) == ["bad|rot|1", "bad|trans|0", "bad|trans|0.5", "good|rot|1", "good|trans|0", "good|trans|0.5"]
    inside = Q.crop_indices(xyz, crop["centre"], 35.0)
    assert crop["n_subset"] == len(inside) == len(crop["labels"])                       # nothing thinned in this call
    rel = xyz[inside].astype(np.float64); rel[:, :2] -= crop["centre"]; rel[:, 2] -= crop["z0"]
    inside = inside[Q.spatial_order(np.round(rel * 100.0).astype(np.int64))]            # the file is in spatial order
    assert np.array_equal(crop["labels"], lab[inside])
    rel = xyz[inside].astype(np.float64); rel[:, :2] -= crop["centre"]; rel[:, 2] -= crop["z0"]
    assert np.abs(crop["xyz"] - rel).max() <= 0.0051 and crop["xyz"][:, 2].min() == 0.0  # centimetres
    assert np.all(np.diff(crop["xyz"][:, 0]) >= 0)
    assert np.array_equal(crop["preds"]["bad|trans|0.5"], preds["noisy"]["bad"][inside])
    assert np.array_equal(crop["preds"]["bad|trans|0"], lab[inside])                    # draw 0 of the clean level, not draw 1 of 0.5
    cm = crop["cm"]["bad|trans|0.5"]
    assert np.allclose(cm, Q.weighted_cm(lab[inside], preds["noisy"]["bad"][inside], w[inside]), rtol=1e-6)
    assert np.allclose(crop["class_points"], np.bincount(lab[inside], weights=w[inside], minlength=11), rtol=1e-6)
    el = np.isin(lab[inside], Q.ELECTRICAL)
    share = (preds["noisy"]["bad"][inside][el] != lab[inside][el]).mean()               # electrical weights are 1 in this scene
    assert abs(Q.wrong_share(cm) - share) < 1e-6 and 0.05 < share < 0.6
    assert Q.wrong_share(crop["cm"]["bad|trans|0"]) == 0.0 and 0 < Q.wrong_share(crop["cm"]["good|rot|1"]) < 0.01
    assert np.isnan(Q.wrong_share(np.zeros((11, 11))))
    # the second pylon is 120 m away: its crop does not hold the mislabelled span around the first one
    other = Q.load_crop(paths[1])
    assert np.linalg.norm(other["centre"] - centres[1]) < 2.0 and Q.wrong_share(other["cm"]["bad|trans|0.5"]) == 0.0


def test_crop_dump_thins_the_points_but_not_the_statistics(tmp_path):
    full, *_ = dump_scene(str(tmp_path / "full"), radius=35.0, pylons=1, voxel=0.0, max_points=10**9)
    thin, *_ = dump_scene(str(tmp_path / "thin"), radius=35.0, pylons=1, voxel=0.25, max_points=3000)
    a, b = Q.load_crop(full.paths("zs")[0]), Q.load_crop(thin.paths("zs")[0])
    assert len(b["labels"]) <= 3000 < len(a["labels"]) and b["n_subset"] == a["n_subset"] == len(a["labels"])
    assert all(np.array_equal(a["cm"][k], b["cm"][k]) for k in a["cm"])                 # same numbers, fewer points to draw
    assert set(np.unique(b["labels"])) >= {0, 1, 2, 3}                                  # every power-line class is still there
    assert b["preds"]["bad|trans|0.5"].shape == b["labels"].shape
    assert thin.settings()["voxel"] == 0.25 and thin.settings()["levels"] == [["rot", 1.0], ["trans", 0.0], ["trans", 0.5]]
    only = Q.CropDump(str(tmp_path / "only"), [("trans", 0.5)], models=["good"], pylons=1)
    xyz, lab, w, centres = synth_scene()
    only(zone="zs", sweep="trans", level=0.5, draw=0, preds=fake_preds(xyz, lab, centres)["noisy"], xyz=xyz, labels=lab)   # no weights: 1
    c = Q.load_crop(only.paths("zs")[0])
    assert sorted(c["preds"]) == ["good|trans|0.5"] and abs(c["cm"]["good|trans|0.5"].sum() - c["n_subset"]) < 0.5


def test_levels_and_keys():
    assert Q.parse_levels("trans:0, trans:0.5,rot:1,") == [("trans", 0.0), ("trans", 0.5), ("rot", 1.0)]
    assert Q.key("concat_n", "trans", 0.50) == "concat_n|trans|0.5" and Q.split_key("a|b|rot|0.25") == ("a|b", "rot", 0.25)


def test_view_across_and_along_the_line():
    xyz, lab, _, centres = synth_scene(line_deg=30.0)
    rel = xyz.astype(np.float64); rel[:, :2] -= centres[0]
    d = Q.line_direction(rel, lab)
    assert abs(abs(d @ [np.cos(np.radians(30)), np.sin(np.radians(30))]) - 1.0) < 1e-3 and d[0] > 0
    cond = lab == 1
    u, v, depth = Q.view_coords(rel, lab)                                               # across the line: the conductors run left-right
    assert np.ptp(u[cond]) > 200 and np.ptp(depth[cond]) < 20 and np.allclose(v, rel[:, 2])
    u, v, depth = Q.view_coords(rel, lab, azimuth=0.0)                                  # along the line: they are seen end-on
    assert np.ptp(u[cond]) < 20 and np.ptp(depth[cond]) > 200
    right, up, towards = Q.view_axes(d, azimuth=60.0, elevation=25.0)                   # an oblique view is still an orthonormal frame
    m = np.stack([right, up, towards])
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-12) and np.allclose(np.cross(right, up), towards) and towards[2] > 0
    assert np.allclose(Q.line_direction(np.zeros((1, 3)), np.zeros(1, np.uint8)), [1.0, 0.0])


def test_raster_layers_and_growth():
    cells, pts = pq.zbuffer(np.array([0, 0, 2]), np.array([0, 0, 1]), np.array([1.0, 5.0, 2.0]), 3)
    assert cells.tolist() == [0, 5] and pts.tolist() == [1, 2]                          # the nearer point of a cell wins
    assert pq.zbuffer(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), 3)[0].size == 0
    img = np.full((5, 5), -1); img[2, 2] = 7
    g1 = pq.grow(img, 1)
    assert (g1 == 7).sum() == 9 and g1[0, 0] == -1 and np.array_equal(pq.grow(img, 0), img)
    assert (pq.grow(img, 2) == 7).sum() == 21                                           # 5 x 5 without its corners
    img[2, 3] = 9
    assert pq.grow(img, 1)[2, 3] == 9 and pq.grow(img, 1)[2, 4] == 9                    # occupied cells keep their own point
    # four points seen across a line along x (the viewer stands at +y): a cable point, a wrongly predicted cable point
    # 20 cm behind it, and two ground points that fix the direction of the line
    crop = {"xyz": np.array([[0.25, 0.0, 5.3], [0.25, -0.2, 5.3], [3.5, 0.0, 0.5], [-3.3, 0.0, 0.5]], np.float32),
            "labels": np.array([1, 1, 6, 6], np.uint8), "preds": {"m|trans|0.5": np.array([1, 5, 6, 6], np.uint8)}}
    scene = pq.Scene(crop, cell=1.0)
    assert (scene.H, scene.W) == (5, 7) and scene.electrical.tolist() == [True, True, False, False]
    assert scene.top[1].tolist() == [0]                                                 # the z-buffer shows the nearer cable point
    rgb, emphasis = pq.panel_colours(scene, crop)
    gt = scene.image(rgb, emphasis, thick=0)
    cable = np.argwhere((gt == pq.PALETTE[1]).all(-1)); ground = np.argwhere((gt == pq.PALETTE[6]).all(-1))
    assert cable.tolist() == [[0, 3]] and ground.tolist() == [[4, 0], [4, 6]]           # the top row of the image is the highest
    assert (gt == 255).all(-1).sum() == 5 * 7 - 3
    rgb, emphasis = pq.panel_colours(scene, crop, "m|trans|0.5")
    err = scene.image(rgb, emphasis, thick=0, thick_wrong=0)
    assert tuple(err[0, 3]) == pq.WRONG_RGB                                             # the wrong point is drawn over the right one in front of it
    assert tuple(err[4, 0]) == pq.BACK_RGB and tuple(err[4, 6]) == pq.BACK_RGB
    assert (scene.image(rgb, emphasis, thick=0, thick_wrong=1) == pq.WRONG_RGB).all(-1).sum() == 6   # grown by one cell (top row: 2 x 3)
    crop["preds"]["m|trans|0.5"] = np.array([1, 1, 2, 6], np.uint8)                     # a ground point is predicted as a cable
    rgb, emphasis = pq.panel_colours(scene, crop, "m|trans|0.5")
    err = scene.image(rgb, emphasis, thick=0, thick_wrong=0)
    assert tuple(err[0, 3]) == pq.RIGHT_RGB and sorted(map(tuple, (err[4, 0], err[4, 6]))) == sorted([pq.ALARM_RGB, pq.BACK_RGB])
    rgb, emphasis = pq.panel_colours(scene, crop, "m|trans|0.5", style="classes")
    assert tuple(rgb[2]) == tuple(pq.PALETTE[2]) and len(emphasis) == 2
    half = pq.Scene(crop, cell=1.0, half_width=1.0)                                     # only the cable points are within 1 m of the centre line
    assert half.idx.tolist() == [0, 1]
    slab = pq.Scene(crop, cell=1.0, slab=0.1)                                           # the point 20 cm behind the view plane is cut
    assert slab.idx.tolist() == [0, 2, 3]


def test_labels_of_models_and_levels():
    groups = [("Feature concatenation", "Delivered poses in training", ["concat_n"]), ("Feature concatenation", "Pose-noise augmentation in training", ["concat_a"])]
    assert pq.model_label("concat_n", groups) == "Feature concatenation,\ndelivered poses in training"
    assert pq.model_label("concat_a", groups) == "Feature concatenation,\npose-noise augmentation"
    assert pq.model_label("lidar", groups) == "LiDAR-only" and pq.model_label("something_else", groups) == "something_else"
    assert pq.model_label("late_a").startswith("Late fusion") and pq.model_label("win5_n_s2").startswith("Window 5×5")   # the names of the paper tables
    assert pq.level_label("rot", 0.0) == pq.level_label("trans", 0.0) == "Delivered poses"
    assert "50 cm" in pq.level_label("trans", 0.5) and "1°" in pq.level_label("rot", 1.0) and "25 cm" in pq.level_label("drift", 0.25)


def test_qualitative_figure_and_list(tmp_path):
    dump, *_ = dump_scene(str(tmp_path / "crops"), radius=35.0, pylons=2)
    crop = Q.load_crop(dump.paths("zs")[0])
    info = pq.figure(crop, ["bad", "good", "absent"], [("trans", 0.0), ("trans", 0.5), ("rot", 1.0)], str(tmp_path / "fig"), dpi=120)
    assert (info["rows"], info["cols"]) == (2, 4) and len(info["panels"]) == 6           # a model that is not in the crop gets no row
    assert info["skipped"] == ["absent|trans|0", "absent|trans|0.5", "absent|rot|1"]
    shares = {(m, s, lv): v for m, s, lv, v in info["panels"]}
    assert shares[("bad", "trans", 0.0)] == 0.0 and shares[("bad", "trans", 0.5)] > 0.05 and shares[("good", "rot", 1.0)] < 0.01
    assert open(tmp_path / "fig.pdf", "rb").read(5) == b"%PDF-" and os.path.getsize(tmp_path / "fig.png") > 10_000
    # the delivered poses are level 0 of any sweep: "rot:0" finds the clean panel that was saved as "trans:0"
    one = pq.figure(crop, ["bad", "good"], [("rot", 1.0)], str(tmp_path / "one"), style="classes", half_width=20.0, dpi=120)
    assert (one["rows"], one["cols"]) == (1, 3) and pq.crop_key(crop, "bad", "rot", 0.0) == "bad|trans|0"
    assert pq.crop_key(crop, "bad", "drift", 0.5) is None
    try:
        pq.figure(crop, ["absent"], [("trans", 0.5)], str(tmp_path / "none"))
        raise AssertionError("a figure without any panel must fail")
    except ValueError as e:
        assert "none of" in str(e)
    lines = []
    rows = pq.list_crops(str(tmp_path / "crops"), out=lines.append)
    assert len(rows) == 2 * 2 * 3 and {r["file"] for r in rows} == {"zs_crop0.npz", "zs_crop1.npz"}
    worst = max(rows, key=lambda r: r["wrong"])
    assert worst["file"] == "zs_crop0.npz" and worst["model"] == "bad" and worst["level"] > 0
    assert any("zone zs" in ln and "subset points" in ln for ln in lines) and any("conductor cable" in ln for ln in lines)
    assert pq.list_crops(str(tmp_path), out=lines.append) == [] and "no crop files" in lines[-1]


def test_figure_rows_one_row_per_pylon(tmp_path):
    dump, *_ = dump_scene(str(tmp_path / "crops"), radius=35.0, pylons=2)
    crops = [Q.load_crop(p) for p in dump.paths("zs")]
    panels = pq.parse_panels("bad:trans:0, bad:rot:1,good:rot:1")
    assert panels == [("bad", "trans", 0.0), ("bad", "rot", 1.0), ("good", "rot", 1.0)]
    info = pq.figure_rows(crops, panels, str(tmp_path / "rows"), dpi=120, half_width=[20.0, 0.0], row_labels=["first", "second"])
    assert (info["rows"], info["cols"]) == (2, 4) and len(info["panels"]) == 6
    assert [p[:4] for p in info["panels"][:3]] == [(0, "bad", "trans", 0.0), (0, "bad", "rot", 1.0), (0, "good", "rot", 1.0)]
    shares = [p[4] for p in info["panels"]]
    assert shares[0] == 0.0 and shares[1] > 0.05 and shares[4] == 0.0                     # the mislabelled span is at the first pylon only
    assert info["raster"][0][1] < info["raster"][1][1]                                    # the first view is the narrower one
    assert open(tmp_path / "rows.pdf", "rb").read(5) == b"%PDF-" and os.path.getsize(tmp_path / "rows.png") > 10_000
    assert pq.figure_rows(crops[:1], panels[:1], str(tmp_path / "single"), dpi=120, style="classes")["cols"] == 2
    for bad in ([("bad", "drift", 0.5)], [("absent", "rot", 1.0)]):                       # a hole in a paper figure must not pass silently
        try:
            pq.figure_rows(crops, bad, str(tmp_path / "hole"))
            raise AssertionError("a missing panel must fail")
        except ValueError as e:
            assert "not in the crops" in str(e)
    try:
        pq.figure_rows(crops, panels, str(tmp_path / "views"), half_width=[20.0])
        raise AssertionError("one view parameter per crop, or one for all")
    except AssertionError as e:
        assert "values for 2 crops" in str(e)
    rows = pq.list_crops(str(tmp_path / "crops"), out=lambda s: None, json_path=str(tmp_path / "crops.json"))
    stats = pq.crop_statistics(str(tmp_path / "crops.json"), "bad", "rot", 1.0)
    assert stats == pq.crop_statistics(rows, "bad", "rot", 1.0) and [f for f, _ in stats] == ["zs_crop1.npz", "zs_crop0.npz"]
    doc = __import__("json").load(open(tmp_path / "crops.json"))
    assert [c["file"] for c in doc["crops"]] == ["zs_crop0.npz", "zs_crop1.npz"] and doc["crops"][0]["points_subset"] >= doc["crops"][0]["points_kept"]
    assert "Conductor cable" in doc["crops"][0]["class_points"] and len(doc["rows"]) == len(rows)


def test_paper_figure_from_the_crops_in_the_repository(tmp_path):
    """The qualitative figure of the paper: two of the twelve pylons of run qual_v1, with the numbers the caption quotes."""
    import statistics
    folder = os.path.join(HERE, "..", "results", "qualitative")
    stats = pq.crop_statistics(os.path.join(folder, "crops.json"))                         # concat_n at 1 deg, all twelve pylons
    assert len(stats) == 12 and len({f.split("_")[0] for f, _ in stats}) == 6
    assert stats[-1][0] == pq.PAPER["crops"][0] == "t1z8_crop0.npz"                        # row (a): the largest share
    assert pq.PAPER["crops"][1] in (stats[5][0], stats[6][0])                              # row (b): one of the two middle ones
    assert [round(100 * v, 1) for _, v in (stats[0], stats[5], stats[6], stats[-1])] == [0.5, 7.0, 7.1, 21.7]
    assert round(100 * statistics.median(v for _, v in stats), 1) == 7.0
    # up to here crops.json was enough. The figure itself needs the two crop files, which hold points of the dataset and
    # are for that reason not in the public repository (scripts/export_public.py): there the test ends here
    if not all(os.path.exists(os.path.join(folder, f)) for f in pq.PAPER["crops"]):
        import pytest
        pytest.skip("the crop files of the figure are not in this repository")
    info = pq.paper_figure(out_dir=str(tmp_path), dpi=150)
    assert [round(100 * p[4], 1) for p in info["panels"]] == [0.9, 21.7, 0.7, 0.4, 7.0, 0.7]
    assert [p[1:4] for p in info["panels"][:3]] == [("concat_n", "trans", 0.0), ("concat_n", "rot", 1.0), ("concat_a", "rot", 1.0)]
    assert open(info["out"] + ".pdf", "rb").read(5) == b"%PDF-" and os.path.getsize(info["out"] + ".png") > 50_000
    crop = Q.load_crop(os.path.join(folder, "t1z8_crop0.npz"))
    assert crop["zone"] == "t1z8" and crop["radius"] == 35.0 and len(crop["preds"]) == 32  # 8 models x 4 levels
    cm = crop["cm"]["concat_n|rot|1"]                                                      # where the cable points go at this pylon
    assert round(100 * cm[1, 5] / cm[1].sum(), 1) == 57.2 and round(100 * cm[2, 5] / cm[2].sum(), 1) == 69.9
    assert round(100 * cm[0, 0] / cm[0].sum(), 1) == 98.1
    assert np.array_equal(crop["preds"]["lidar|trans|0"], crop["preds"]["lidar|rot|1"])    # LiDAR-only does not read the cameras
