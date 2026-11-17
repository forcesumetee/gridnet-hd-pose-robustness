"""Model-free label transfer under pose error: counting, error models, seeds and the zone script (numpy only)."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

from gridnet import label_transfer as lt
from gridnet.camera import Calibration, Pose, perturb_pose_gaussian, project, rot_from_opk, visible

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CAL = Calibration(width=400, height=300, f=300.0, k1=-0.05, k2=0.01)
ORIGIN = np.array([500000.0, 5100000.0, 400.0])


def scene(seed=0, n=120_000):
    """Herbaceous ground (class 6) with a conductor (class 1) 20 m above it, in cache coordinates (float32)."""
    rng = np.random.default_rng(seed)
    ground = np.column_stack([rng.uniform(-60, 60, n), rng.uniform(-50, 50, n), rng.uniform(0, 0.05, n)])
    m = 6000
    cable = np.column_stack([rng.uniform(-60, 60, m), rng.normal(2.0, 0.01, m), 20 + rng.normal(0, 0.01, m)])
    xyz = np.concatenate([ground, cable]).astype(np.float32)
    labels = np.concatenate([np.full(n, 6, np.uint8), np.full(m, 1, np.uint8)])
    labels[:200] = 255                                    # a few unlabelled points
    return xyz, labels


def poses():
    """Three nadir cameras 45 m above the ground, in cache coordinates."""
    return {f"img{i}": Pose(S=np.array([dx, dy, 45.0]), R=rot_from_opk(0.0, 0.0, kappa), opk=np.array([0.0, 0.0, kappa]))
            for i, (dx, dy, kappa) in enumerate([(-20.0, 0.0, 0.0), (0.0, 5.0, 30.0), (25.0, -5.0, 90.0)])}


def render_mask(xyz, labels, pose, cal=CAL):
    """The mask an annotator would deliver: a closed ground surface (the sparse points painted 7x7) with the thin
    class painted 3x3 on top; what no point reaches stays unlabelled."""
    u, v, z, ok = project(xyz.astype(np.float64), pose, cal)
    vis, _ = visible(u, v, z, ok, cal)
    vis &= labels != 255
    ui, vi, L = u[vis].astype(int), v[vis].astype(int), labels[vis]
    mask = np.full((cal.height, cal.width), 255, np.uint8)
    for thin, r in ((False, 3), (True, 1)):                # every pixel of the thin class is painted after the ground
        k = (L == 1) == thin
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                mask[np.clip(vi[k] + dv, 0, cal.height - 1), np.clip(ui[k] + du, 0, cal.width - 1)] = L[k]
    return mask


def load_script():
    spec = importlib.util.spec_from_file_location("label_transfer_zone", os.path.join(ROOT, "scripts", "label_transfer_zone.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["label_transfer_zone"] = mod               # the worker pool pickles its task function by module name
    spec.loader.exec_module(mod)
    return mod


def test_gaussian_pose_noise_is_the_protocol_model():
    pose = poses()["img1"]
    rng = np.random.default_rng(0)
    q = [perturb_pose_gaussian(pose, rng, sigma_t=0.10, sigma_r_deg=0.5) for _ in range(4000)]
    dS = np.array([p.S - pose.S for p in q])
    assert np.allclose(dS.std(0), 0.10, rtol=0.05) and abs(dS.mean()) < 0.01           # N(0, sigma^2) per axis
    ang = np.array([np.degrees(np.arccos(np.clip((np.trace(p.R @ pose.R.T) - 1) / 2, -1, 1))) for p in q])
    assert abs(np.sqrt((ang ** 2).mean()) - 0.5 * np.sqrt(3)) < 0.03                    # |w|, w ~ N(0, sigma^2 I_3)
    assert all(np.allclose(p.R @ p.R.T, np.eye(3), atol=1e-12) for p in q[:20])
    same = perturb_pose_gaussian(pose, rng)
    assert same.S is pose.S and same.R is pose.R
    # the two error models of `perturbed`: sigma per axis against a fixed length; level 0 is the delivered pose
    a = lt.perturbed(pose, "translation_cm", 10.0, np.random.default_rng(1), "fixed")
    assert abs(np.linalg.norm(a.S - pose.S) - 0.10) < 1e-12 and a.R is pose.R
    b = lt.perturbed(pose, "rotation_deg", 0.5, np.random.default_rng(1), "protocol")
    assert b.S is pose.S and not np.allclose(b.R, pose.R)
    assert lt.perturbed(pose, "none", 0.0, rng) is pose and lt.perturbed(pose, "rotation_deg", 0.0, rng) is pose
    with pytest.raises(ValueError):
        lt.perturbed(pose, "rotation_deg", 0.5, rng, "other")


def test_delivered_pose_returns_the_labels_and_the_thin_class_fails_first():
    xyz, labels = scene()
    pose = poses()["img1"]
    mask = render_mask(xyz, labels, pose)
    cand = lt.candidates(xyz, pose, CAL)
    hits, total, ha, na = lt.transfer_counts(xyz[cand].astype(np.float64), labels[cand], mask, pose, CAL)
    assert total[1] > 500 and total[6] > 20_000 and total[[0, 2, 3, 4, 5, 7, 8, 9, 10]].sum() == 0
    assert hits[1] == total[1] and hits[6] / total[6] > 0.97 and ha / na > 0.97         # ground under the cable reads "cable"
    # the candidate set changes nothing: the same counts from all points of the scene
    h2, t2, ha2, na2 = lt.transfer_counts(xyz.astype(np.float64), labels, mask, pose, CAL)
    assert np.array_equal(hits, h2) and np.array_equal(total, t2) and (ha, na) == (ha2, na2)
    # ... also for a strongly perturbed pose (what the camera sees then is inside the candidates as well)
    far = perturb_pose_gaussian(pose, np.random.default_rng(3), sigma_t=1.0, sigma_r_deg=1.0)
    a = lt.transfer_counts(xyz[cand].astype(np.float64), labels[cand], mask, far, CAL)
    b = lt.transfer_counts(xyz.astype(np.float64), labels, mask, far, CAL)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
    # one metre of position error: the cable is lost, the ground hardly notices
    recs = lt.image_records(xyz, labels, mask, pose, CAL, "img1", trans_cm=[100.0], rot_deg=[], draws=8, noise="fixed")
    assert [r["kind"] for r in recs] == ["none"] + ["translation_cm"] * 8 and recs[0]["hits"][1] == recs[0]["total"][1]
    rows = lt.summarize(recs, min_points=1)
    assert [(r["perturbation"], r["level"]) for r in rows] == [("none", 0.0), ("translation_cm", 100.0)]
    assert rows[0]["conductor"] == 1.0 and rows[1]["conductor"] < 0.5 and rows[1]["herb"] > 0.95 and rows[1]["all"] > 0.9
    assert np.isnan(rows[1]["pylon"]) and rows[1]["draws"] == 8 and rows[1]["images"] == 1


def test_draws_depend_on_the_image_name_and_level_only():
    a = lt.draw_rng(0, "img1", "rotation_deg", 0.1, 0).normal(size=3)
    assert np.array_equal(a, lt.draw_rng(0, "img1", "rotation_deg", 0.1, 0).normal(size=3))
    others = [lt.draw_rng(*k).normal(size=3) for k in ((1, "img1", "rotation_deg", 0.1, 0), (0, "img2", "rotation_deg", 0.1, 0),
                                                        (0, "img1", "translation_cm", 0.1, 0), (0, "img1", "rotation_deg", 0.25, 0),
                                                        (0, "img1", "rotation_deg", 0.1, 1))]
    assert all(not np.array_equal(a, o) for o in others)


def test_summarize_pools_points_not_ratios():
    def rec(image, kind, level, draw, hits1, total1, ha, na):
        h, t = [0] * 11, [0] * 11
        h[1], t[1] = hits1, total1
        return {"image": image, "kind": kind, "level": level, "draw": draw, "hits": h, "total": t, "hits_all": ha, "n_all": na}

    recs = [rec("a", "none", 0.0, 0, 100, 100, 990, 1000), rec("b", "none", 0.0, 0, 300, 300, 2970, 3000),
            rec("a", "rotation_deg", 0.5, 0, 10, 100, 900, 1000), rec("a", "rotation_deg", 0.5, 1, 30, 100, 900, 1000),
            rec("b", "rotation_deg", 0.5, 0, 240, 300, 2400, 3000), rec("b", "rotation_deg", 0.5, 1, 240, 300, 2400, 3000),
            rec("c", "rotation_deg", 0.5, 0, 0, 10, 50, 100)]                              # an image with few cable points
    rows = lt.summarize(recs, min_points=50)
    r = rows[1]
    assert rows[0]["conductor"] == 1.0 and rows[0]["all"] == 0.99 and rows[0]["images"] == 2
    assert abs(r["conductor"] - 520 / 810) < 1e-12 and abs(r["all"] - 6650 / 8100) < 1e-12        # sums of points
    assert r["images"] == 3 and r["draws"] == 2 and abs(r["n"] - 8100 / 5) < 1e-9
    assert abs(r["conductor_q25"] - 0.35) < 1e-12 and abs(r["conductor_q75"] - 0.65) < 1e-12      # images a (0.2) and b (0.8); c has too few points
    assert abs(r["all_q25"] - 0.65) < 1e-12 and np.isnan(r["pylon_q25"])                           # a 0.9, b 0.8, c 0.5


def write_zone(tmp_path, zone="zz"):
    xyz, labels = scene()
    cache, data = str(tmp_path / "cache"), str(tmp_path / "data")
    os.makedirs(os.path.join(cache, zone)); os.makedirs(os.path.join(data, zone, "pose")); os.makedirs(os.path.join(data, zone, "masks"))
    np.savez(os.path.join(cache, zone, "points.npz"), xyz=xyz, origin=ORIGIN, labels=labels)
    open(os.path.join(data, zone, "pose", "camera_calibration.xml"), "w").write(
        f"<calibration><width>{CAL.width}</width><height>{CAL.height}</height><f>{CAL.f}</f><k1>{CAL.k1}</k1><k2>{CAL.k2}</k2></calibration>")
    lines = ["# PhotoID X Y Z Omega Phi Kappa r11..r33"]
    for pid, p in poses().items():
        lines.append(pid + " " + " ".join(f"{v:.9f}" for v in [*(p.S + ORIGIN), *p.opk, *p.R.ravel()]))
        Image.fromarray(render_mask(xyz, labels, p)).save(os.path.join(data, zone, "masks", pid + ".png"))
    lines.append("img9 " + " ".join(f"{v:.9f}" for v in [*ORIGIN, 0, 0, 0, *np.eye(3).ravel()]))          # a pose without a mask
    open(os.path.join(data, zone, "pose", "cameras_pose.txt"), "w").write("\n".join(lines) + "\n")
    return cache, data


def test_zone_script_resume_subsets_and_pooling(tmp_path, monkeypatch, capsys):
    mod = load_script()
    cache, data = write_zone(tmp_path)
    out = str(tmp_path / "out")

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["label_transfer_zone.py", "--cache", cache, "--data", data, "--zone", "zz", "--out", out,
                                          "--trans", "10,100", "--rot", "1", "--workers", "1", *extra])
        mod.main()
        return capsys.readouterr().out

    log = run("--images", "img1")
    assert "1 images (3 with a mask), 0 done before, 1 to do" in log and "wrote" in log
    first = mod.read_records(os.path.join(out, "zz", "records.jsonl"))["img1"]
    assert len(first) == 1 + 3 and first[0]["kind"] == "none" and first[0]["hits"][1] == first[0]["total"][1] > 500
    log = run()                                                         # all images: img1 is not computed again
    assert "3 images (3 with a mask), 1 done before, 2 to do" in log
    done = mod.read_records(os.path.join(out, "zz", "records.jsonl"))
    assert sorted(done) == ["img0", "img1", "img2"] and done["img1"] == first
    doc = json.load(open(os.path.join(out, "zz", "label_transfer.json")))
    assert doc["images"] == 3 and doc["image_ids"] == ["img0", "img1", "img2"] and doc["settings"]["noise"] == "protocol"
    rows = {(r["perturbation"], r["level"]): r for r in doc["rows"]}
    assert rows[("none", 0.0)]["conductor"] == 1.0 and rows[("translation_cm", 100.0)]["conductor"] < 0.6 < rows[("translation_cm", 100.0)]["herb"]
    # the CSV is what the JSON says, and the figure script reads it
    import csv
    table = list(csv.DictReader(open(os.path.join(out, "zz", "label_transfer.csv"))))
    assert [t["perturbation"] for t in table] == ["none", "translation_cm", "translation_cm", "rotation_deg"]
    assert all(t["noise"] == "protocol" for t in table) and abs(float(table[2]["conductor"]) - rows[("translation_cm", 100.0)]["conductor"]) < 1e-12
    # a worker pool (forked processes: Linux only, as on Colab) gives the same records - seeds per image, level and draw
    out2 = str(tmp_path / "out2")
    monkeypatch.setattr(sys, "argv", ["label_transfer_zone.py", "--cache", cache, "--data", data, "--zone", "zz", "--out", out2,
                                      "--trans", "10,100", "--rot", "1", "--workers", "2" if sys.platform.startswith("linux") else "1"])
    mod.main(); capsys.readouterr()
    assert mod.read_records(os.path.join(out2, "zz", "records.jsonl")) == done
    # other settings in the same folder are refused; --force starts again
    with pytest.raises(SystemExit, match="other settings"):
        run("--noise", "fixed")
    assert "0 done before, 3 to do" in run("--noise", "fixed", "--force")
    with pytest.raises(SystemExit, match="no mask or pose"):
        run("--noise", "fixed", "--images", "img9")
    # an interrupted run leaves a truncated line: it is ignored and the image is computed again
    with open(os.path.join(out2, "zz", "records.jsonl"), "a") as fh:
        fh.write('{"image": "imgX", "records": [')
    assert sorted(mod.read_records(os.path.join(out2, "zz", "records.jsonl"))) == ["img0", "img1", "img2"]
    # pooling of zones: the same zone twice doubles every count and leaves the shares
    os.rename(os.path.join(out2, "zz"), os.path.join(out2, "za"))
    s = json.load(open(os.path.join(out2, "za", "settings.json"))); s["zone"] = "za"
    json.dump(s, open(os.path.join(out2, "za", "settings.json"), "w"))
    import shutil
    shutil.copytree(os.path.join(out2, "za"), os.path.join(out2, "zb"))
    s["zone"] = "zb"; json.dump(s, open(os.path.join(out2, "zb", "settings.json"), "w"))
    monkeypatch.setattr(sys, "argv", ["label_transfer_zone.py", "--out", out2, "--combine", "za,zb"])
    mod.main()
    assert "2 zones, 6 images" in capsys.readouterr().out
    pooled = json.load(open(os.path.join(out2, "all", "label_transfer.json")))
    one = json.load(open(os.path.join(out2, "za", "label_transfer.json")))
    for a, b in zip(pooled["rows"], one["rows"]):
        assert a["images"] == 2 * b["images"] and abs(a["conductor"] - b["conductor"]) < 1e-12 and abs(a["all"] - b["all"]) < 1e-12
