"""Leaderboard files: voxel predictions -> one class per LAS point in file order (pure numpy; the LAS tests need laspy)."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

from gridnet import submission as S
from gridnet.classes import REMAP
from gridnet.preprocess.geometry import majority_labels, voxel_grid
from gridnet.train.data import ConfusionMeter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def fake_cache(root, zone, n=6000, seed=0, labeled=True):
    """A points.npz with the arrays the export needs (inverse, keep, labels) from a real voxel grid. Returns
    (raw labels in the 11-class space, voxel labels, keep, inverse)."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 3.0, (n, 3))
    keep, inverse = voxel_grid(xyz, 0.25)
    raw = rng.choice([0, 1, 4, 6, 8, 255], size=n, p=[0.1, 0.1, 0.3, 0.3, 0.1, 0.1]).astype(np.uint8) if labeled else np.full(n, 255, np.uint8)
    vox = majority_labels(raw, inverse, len(keep))
    os.makedirs(os.path.join(root, zone), exist_ok=True)
    np.savez(os.path.join(root, zone, "points.npz"), xyz=xyz[keep].astype(np.float32), labels=vox, inverse=inverse.astype(np.int32),
             keep=keep.astype(np.int32), labeled=np.array(labeled))
    return raw, vox, keep, inverse


def test_raw_order_and_its_checks():
    rng = np.random.default_rng(1)
    xyz = rng.uniform(0, 2.0, (5000, 3))
    keep, inverse = voxel_grid(xyz, 0.2)
    pred = rng.integers(0, 11, len(keep)).astype(np.int64)
    raw = S.to_raw_order(pred, inverse, keep)
    assert raw.dtype == np.uint8 and raw.shape == (5000,)
    assert np.array_equal(raw[keep], pred)                                   # the representative point has its voxel's class
    assert all(len(set(raw[inverse == v])) == 1 for v in range(0, len(keep), 37))   # one class per voxel
    with pytest.raises(ValueError, match="not from the same cache"):
        S.to_raw_order(pred[:-1], inverse, None)                             # prediction of another zone / another cache
    with pytest.raises(ValueError, match="outside 0..10"):
        S.to_raw_order(np.where(np.arange(len(pred)) == 3, 11, pred), inverse, keep)
    with pytest.raises(ValueError, match="identity"):
        S.to_raw_order(pred, inverse, keep[::-1].copy())
    with pytest.raises(ValueError, match="one class per voxel"):
        S.to_raw_order(np.zeros((len(keep), 11)), inverse, keep)             # votes instead of classes


def test_file_format_is_the_leaderboards(tmp_path):
    data = np.random.default_rng(2).integers(0, 11, 1234)
    path = S.write_submission(str(tmp_path / "sub" / "t1z4.npz"), data)
    assert os.listdir(tmp_path / "sub") == ["t1z4.npz"]                      # no temporary file left
    with np.load(path) as f:                                                 # what the leaderboard reads
        assert f.files == ["data"] and f["data"].dtype == np.uint8 and np.array_equal(f["data"], data)
    info = S.check_submission(path, 1234)
    assert info["points"] == 1234 and sum(info["counts"]) == 1234 and len(info["counts"]) == 11 and info["bytes"] > 0
    with pytest.raises(ValueError, match="1,234 predictions for 1,235 LAS points"):
        S.check_submission(path, 1235)
    np.savez_compressed(str(tmp_path / "bad.npz"), classification=data.astype(np.uint8))
    with pytest.raises(ValueError, match="exactly the array 'data'"):
        S.check_submission(str(tmp_path / "bad.npz"), 1234)
    np.savez_compressed(str(tmp_path / "bad2.npz"), data=np.full(10, 21, np.uint8))    # original (un-remapped) class ids
    with pytest.raises(ValueError, match="class id 21"):
        S.check_submission(str(tmp_path / "bad2.npz"), 10)
    assert S.TEST_ZONES == ["t1z4", "t1z5a", "t1z7", "t3z1", "t3z2", "t3z5", "t5a2", "t6z1", "t6z5"]


def test_metrics_equal_the_evaluation_code():
    rng = np.random.default_rng(3)
    gt = rng.choice([0, 1, 4, 6, 255], size=20000).astype(np.uint8)
    pred = rng.choice([0, 1, 4, 6, 9], size=20000).astype(np.uint8)         # class 9 never occurs in the ground truth
    cm = S.confusion(gt, pred, chunk=3000)
    meter = ConfusionMeter(); meter.add(gt, pred)
    assert np.array_equal(cm, meter.m)
    m = S.metrics(cm)
    assert abs(m["mIoU"] - 100 * meter.miou()) < 1e-9 and abs(m["mIoU_present"] - 100 * meter.miou(present_only=True)) < 1e-9
    assert abs(m["OA"] - 100 * meter.oa()) < 1e-9 and m["per_class"]["road"] is None and m["per_class"]["water"] == 0.0
    assert m["mIoU_present"] > m["mIoU"] and m["points"] == int((gt != 255).sum())
    empty = S.metrics(np.zeros((11, 11), np.int64))
    assert empty["mIoU"] is None and empty["mIoU_present"] is None and empty["OA"] == 0.0


def test_export_script_end_to_end(tmp_path, monkeypatch, capsys):
    cache, preds, out = str(tmp_path / "cache"), str(tmp_path / "pred"), str(tmp_path / "out")
    raw_a, vox_a, keep_a, inv_a = fake_cache(cache, "t1z4", labeled=False)
    raw_b, vox_b, keep_b, inv_b = fake_cache(cache, "valz", seed=5)
    rng = np.random.default_rng(4)
    pa = rng.integers(0, 11, len(keep_a)).astype(np.uint8)
    pb = np.where(vox_b == 255, 6, vox_b).astype(np.uint8); pb[::7] = 4       # mostly right
    for z, p in (("t1z4", pa), ("valz", pb)):
        os.makedirs(os.path.join(preds, z)); np.save(os.path.join(preds, z, "fused_pred.npy"), p)
    mod = load_script("export_submission")
    argv = ["export_submission.py", "--cache", cache, "--pred", os.path.join(preds, "{zone}", "fused_pred.npy"), "--out", out,
            "--zones", "t1z4,valz", "--name", "late_a"]
    monkeypatch.setattr(sys, "argv", argv); mod.main()
    text = capsys.readouterr().out
    da, db = S.read_submission(os.path.join(out, "t1z4.npz")), S.read_submission(os.path.join(out, "valz.npz"))
    assert len(da) == len(raw_a) == 6000 and np.array_equal(da, pa[inv_a]) and np.array_equal(db[keep_b], pb)
    man = json.load(open(os.path.join(out, "manifest.json")))
    assert man["name"] == "late_a" and set(man["zones"]) == {"t1z4", "valz"} and man["class_names"][9] == "water"
    assert "voxel_metrics" not in man["zones"]["t1z4"] and man["zones"]["t1z4"]["points"] == 6000       # unlabeled zone: no score
    vm = man["zones"]["valz"]["voxel_metrics"]
    meter = ConfusionMeter(); meter.add(vox_b, pb)
    assert abs(vm["mIoU_present"] - 100 * meter.miou(present_only=True)) < 1e-9 and "per voxel" in text
    assert "raw_metrics" not in man["zones"]["valz"]                          # no LAS given
    monkeypatch.setattr(sys, "argv", argv); mod.main()                        # second run: nothing rewritten, manifest kept
    assert capsys.readouterr().out.count("exists (skipped") == 2
    assert set(json.load(open(os.path.join(out, "manifest.json")))["zones"]) == {"t1z4", "valz"}
    np.save(os.path.join(preds, "t1z4", "fused_pred.npy"), (pa + 1) % 11)
    monkeypatch.setattr(sys, "argv", argv + ["--force"]); mod.main()
    assert np.array_equal(S.read_submission(os.path.join(out, "t1z4.npz")), ((pa + 1) % 11)[inv_a])
    monkeypatch.setattr(sys, "argv", argv[:-4] + ["--zones", "t1z4,t9z9", "--force"])      # a zone without a prediction: reported, exit 1
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 1 and "NO prediction for ['t9z9']" in capsys.readouterr().out
    np.save(os.path.join(preds, "valz", "fused_pred.npy"), pb[:-5])           # a prediction that does not belong to the cache
    monkeypatch.setattr(sys, "argv", argv + ["--force"])
    with pytest.raises(ValueError, match="not from the same cache"):
        mod.main()


# ----------------------------------------------------------------------------- with real LAS files
def write_las(path, xyz, ground_truth=None, classification=None):
    laspy = pytest.importorskip("laspy")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(0); header.scales = np.array([0.001, 0.001, 0.001])
    if ground_truth is not None:
        header.add_extra_dim(laspy.ExtraBytesParams(name="ground_truth", type=np.uint8))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if ground_truth is not None:
        las.ground_truth = ground_truth
    if classification is not None:
        las.classification = classification
    os.makedirs(os.path.dirname(path), exist_ok=True)
    las.write(path)


def las_zones(root, n=30000):
    """Two zones on disk: 'valz' with the ground_truth field (original 22-class ids), 't1z4' without it but with a
    filled standard classification field (which must never be taken for labels)."""
    rng = np.random.default_rng(7)
    xyz = np.column_stack([rng.uniform(0, 12, n), rng.uniform(0, 12, n), rng.uniform(0, 0.3, n)]) + np.array([500000.0, 5100000.0, 400.0])
    xyz[: n // 20, 2] += rng.uniform(3, 9, n // 20)
    gt = rng.choice([0, 5, 6, 10, 13, 14, 16, 18, 19, 255], size=n).astype(np.uint8)        # 13 = signage -> unlabeled
    write_las(os.path.join(root, "valz", "lidar", "valz.las"), xyz, ground_truth=gt)
    write_las(os.path.join(root, "t1z4", "lidar", "t1z4.las"), xyz, classification=np.full(n, 2, np.uint8))
    return xyz, gt


def test_a_las_without_ground_truth_is_an_unlabeled_zone(tmp_path):
    pytest.importorskip("laspy")
    from gridnet.io import las_point_count, load_zone_points, read_las_labels
    root = str(tmp_path)
    xyz, gt = las_zones(root)
    val = load_zone_points(os.path.join(root, "valz"))
    assert bool(val["labeled"]) and np.array_equal(val["labels"], REMAP[gt]) and np.allclose(val["xyz"], xyz, atol=2e-3)
    test = load_zone_points(os.path.join(root, "t1z4"))
    assert not bool(test["labeled"]) and (test["labels"] == 255).all() and (test["labels_raw"] == 255).all()
    again = load_zone_points(os.path.join(root, "t1z4"))                      # from the .npz written next to the LAS
    assert not bool(again["labeled"]) and (again["labels"] == 255).all() and np.array_equal(again["xyz"], test["xyz"])
    assert las_point_count(os.path.join(root, "t1z4", "lidar", "t1z4.las")) == len(xyz)
    assert read_las_labels(os.path.join(root, "t1z4", "lidar", "t1z4.las")) is None
    assert np.array_equal(read_las_labels(os.path.join(root, "valz", "lidar", "valz.las"), chunk=7000), REMAP[gt])


def test_cache_then_export_scored_on_the_raw_points(tmp_path, monkeypatch, capsys):
    pytest.importorskip("laspy")
    root, cache, out = str(tmp_path / "data"), str(tmp_path / "cache"), str(tmp_path / "out")
    xyz, gt = las_zones(root)
    build = load_script("build_cache")
    for z in ("valz", "t1z4"):
        monkeypatch.setattr(sys, "argv", ["build_cache.py", "--zone", os.path.join(root, z), "--out", cache, "--no-assoc"])
        build.main()
    log = capsys.readouterr().out
    assert "no ground_truth field" in log and "unlabeled test zone" in open(os.path.join(cache, "t1z4", "report.txt")).read()
    pv, pt = np.load(os.path.join(cache, "valz", "points.npz")), np.load(os.path.join(cache, "t1z4", "points.npz"))
    assert bool(pv["labeled"]) and not bool(pt["labeled"]) and (pt["labels"] == 255).all() and (pv["labels"] != 255).any()
    assert len(pt["inverse"]) == len(xyz) and np.array_equal(pt["inverse"], pv["inverse"])       # same points, same grid
    assert np.array_equal(pt["xyz"], pv["xyz"]) and np.array_equal(pt["hag"], pv["hag"])           # features do not depend on labels
    rng = np.random.default_rng(8)
    pred = np.where(rng.random(len(pv["labels"])) < 0.8, np.where(pv["labels"] == 255, 6, pv["labels"]), 4).astype(np.uint8)
    for z in ("valz", "t1z4"):
        os.makedirs(os.path.join(out, "pred", z)); np.save(os.path.join(out, "pred", z, "fused_pred.npy"), pred)
    mod = load_script("export_submission")
    monkeypatch.setattr(sys, "argv", ["export_submission.py", "--cache", cache, "--pred", os.path.join(out, "pred", "{zone}", "fused_pred.npy"),
                                      "--out", os.path.join(out, "fused"), "--zones", "valz,t1z4", "--las-root", root])
    mod.main()
    man = json.load(open(os.path.join(out, "fused", "manifest.json")))["zones"]
    raw = S.read_submission(os.path.join(out, "fused", "valz.npz"))
    assert len(raw) == len(xyz) and man["valz"]["las_points_checked"] and man["t1z4"]["las_points_checked"]
    meter = ConfusionMeter(); meter.add(REMAP[gt], raw)                       # scored against the labels of the LAS, point by point
    assert abs(man["valz"]["raw_metrics"]["mIoU_present"] - 100 * meter.miou(present_only=True)) < 1e-9
    assert man["valz"]["raw_metrics"]["points"] == int((REMAP[gt] != 255).sum()) > man["valz"]["voxel_metrics"]["points"]
    assert "raw_metrics" not in man["t1z4"] and "voxel_metrics" not in man["t1z4"]
    assert np.array_equal(S.read_submission(os.path.join(out, "fused", "t1z4.npz")), raw)
    # a cache built from another LAS than the one in --las-root is refused
    write_las(os.path.join(root, "t1z4", "lidar", "t1z4.las"), xyz[:-3], classification=np.full(len(xyz) - 3, 2, np.uint8))
    monkeypatch.setattr(sys, "argv", ["export_submission.py", "--cache", cache, "--pred", os.path.join(out, "pred", "{zone}", "fused_pred.npy"),
                                      "--out", os.path.join(out, "fused2"), "--zones", "t1z4", "--las-root", root])
    with pytest.raises(ValueError, match="the LAS has 29,997 points, the cache was built from 30,000"):
        mod.main()
