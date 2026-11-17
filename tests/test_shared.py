"""scripts/compare_shared.py: a run with shared pose errors next to the run with independent ones (pure Python: runs
on the torch-less Mac venv too)."""
import importlib.util
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CLASSES = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb", "rock_soil", "road", "water", "building"]


def load():
    spec = importlib.util.spec_from_file_location("compare_shared", os.path.join(ROOT, "scripts", "compare_shared.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def summary(models, sweeps, value, std=0.0, draws=10):
    """A protocol summary: IoU of every class = value(model, sweep, level, class index) in %, the same for the two
    class groups (so the group means are easy to predict)."""
    curves = []
    for m in models:
        for sw, levels in sweeps.items():
            for lv in levels:
                per = [value(m, sw, lv, c) for c in range(11)]
                for c, name in enumerate(CLASSES):
                    curves.append({"model": m, "sweep": sw, "level": lv, "class": name, "iou_mean": per[c] / 100, "iou_std": std / 100, "draws": draws})
                for g, idx in (("mean_all", range(11)), ("mean_electrical", range(4))):
                    curves.append({"model": m, "sweep": sw, "level": lv, "class": g, "iou_mean": sum(per[i] for i in idx) / len(idx) / 100,
                                   "iou_std": std / 100, "draws": draws})
    disp = [{"sweep": sw, "level": lv, "n": 6, "shift_px": 60.0 * lv, "shift_px_p90": 120.0 * lv, "shift_px_electrical": 90.0 * lv, "coverage": 1.0}
            for sw, levels in sweeps.items() for lv in levels]
    return {"meta": {"models": list(models), "sweeps": sweeps, "draws": draws, "mode": "exact", "views": 8, "zones": ["za", "zb"]},
            "curves": curves, "scores": [], "displacement": disp}


def base(model):
    return {"lidar": 70.0, "imagevote": 60.0}.get(model, 80.0 + (2.0 if model.endswith("_s1") else 0.0))


def test_shared_against_independent_errors():
    cs = load()
    iid_models = ["lidar", "imagevote", "late_nat_b1", "late_n_s1", "late_n_s2", "concat_n", "late_a", "late_a_s1", "concat_a10"]
    sh_models = ["lidar", "imagevote", "late_nat_b1", "late_n_s1", "late_a", "win3_a"]              # win3_a is not in the i.i.d. run
    lv_t, lv_r = [0.0, 0.1, 0.5], [0.0, 0.1, 1.0]

    def iid_value(m, sw, lv, c):                       # independent errors: 10 points per unit of level, LiDAR-only constant
        return base(m) - (0.0 if m == "lidar" else 10.0 * lv)

    def shared_value(m, sw, lv, c):                    # shared errors: twice the loss, and 5 more on class 8 (road)
        return base(m) - (0.0 if m == "lidar" else 20.0 * lv + (5.0 if c == 8 and lv > 0 else 0.0))
    iid = summary(iid_models, {"trans": lv_t, "rot": lv_r, "drift": [0.0, 0.25]}, iid_value, std=0.2)
    sh = summary(sh_models, {"trans_shared": lv_t + [0.25], "rot_shared": lv_r}, shared_value, std=1.5)

    groups = cs.model_groups(sh, iid)                  # head types with the seeds BOTH runs hold; win3_a has none
    assert [(b, l, n) for b, l, n in groups] == [
        ("", "LiDAR-only", ["lidar"]), ("", "ImageVote", ["imagevote"]),
        ("Delivered poses in training", "Late fusion", ["late_nat_b1", "late_n_s1"]),
        ("Pose-noise augmentation in training", "Late fusion", ["late_a"])]
    assert cs.clean_check(sh, iid) < 1e-9
    rows = cs.shared_rows(sh, iid)
    get = {(r["head"], r["block"], r["sweep"], r["level"], r["class"]): r for r in rows}
    late = get[("Late fusion", "Delivered poses in training", "trans_shared", 0.5, "mean_electrical")]
    assert late["n"] == 2 and late["iid_sweep"] == "trans" and late["draws_shared"] == 10
    assert abs(late["iid"] - (81.0 - 5.0)) < 1e-9 and abs(late["shared"] - (81.0 - 10.0)) < 1e-9     # mean of the seeds 80 and 82
    assert abs(late["shared_std_draws"] - 1.5) < 1e-9 and abs(late["iid_std_draws"] - 0.2) < 1e-9
    road = get[("Late fusion", "Delivered poses in training", "rot_shared", 1.0, "road")]
    assert abs((road["shared"] - road["iid"]) - (-10.0 - 5.0)) < 1e-9
    allc = get[("Late fusion", "Pose-noise augmentation in training", "rot_shared", 1.0, "mean_all")]
    assert allc["n"] == 1 and abs(allc["shared"] - (80.0 - 20.0 - 5.0 / 11)) < 1e-9
    assert ("LiDAR-only", "", "trans_shared", 0.25, "mean_all") not in get                           # a level the i.i.d. run does not have
    assert {r["level"] for r in rows if r["sweep"] == "trans_shared"} == {0.0, 0.1, 0.5}
    lines = []
    out = cs.report(sh, iid, out=lines.append)
    text = "\n".join(lines)
    assert out == rows and "largest difference between the runs over the models they share: 0.000" in text and "as it must be" in text
    assert "translation error, mean IoU over the electrical classes" in text and "rotation error, mean IoU over all classes" in text
    row = [l for l in lines if l.strip().startswith("Late fusion") and " 2  " in l][0]               # delivered poses, electrical, translation
    assert "80.0 |  79.0 +- 1.5" in row and "76.0 |  71.0 +- 1.5" in row                             # 10 cm and 50 cm
    assert "rotation error of 1 deg, per class: shared minus independent" in text
    assert any("-15.0" in l and l.count("-10.0") == 10 for l in lines if l.strip().startswith("ImageVote"))
    assert "50 cm: independent 30.0 px, shared 30.0 px" in text and "(electrical classes 45.0 / 45.0)" in text

    # another set of heads behind the same names shows at the delivered poses
    other = summary(sh_models, {"trans_shared": lv_t, "rot_shared": lv_r}, lambda m, sw, lv, c: shared_value(m, sw, lv, c) + (0.4 if m == "late_a" else 0.0))
    assert abs(cs.clean_check(other, iid) - 0.4) < 1e-9
    lines2 = []; cs.report(other, iid, out=lines2.append)
    assert any("NOT the same clean numbers" in l for l in lines2)
    # a run without a common model
    none = summary(["win3_a"], {"trans_shared": lv_t}, shared_value)
    assert cs.clean_check(none, iid) is None and cs.shared_rows(none, iid) == []


def load_tables():
    spec = importlib.util.spec_from_file_location("make_tables", os.path.join(ROOT, "scripts", "make_tables.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_shared_table_of_the_paper():
    """Table "shared" (scripts/make_tables.py): independent | shared side by side, mean over the seeds both runs hold."""
    import json
    import pytest
    mt = load_tables()
    models = ["lidar", "imagevote", "late_nat_b1", "late_n_s1", "late_a", "concat_n"]
    lv_t, lv_r = [0.0, 0.02, 0.1, 0.5], [0.0, 0.1, 1.0]

    def iid_value(m, sw, lv, c):                       # independent: 10 points per unit of level; electrical classes 5 higher
        return base(m) + (5.0 if c < 4 else 0.0) - (0.0 if m == "lidar" else 10.0 * lv)

    def shared_value(m, sw, lv, c):                    # shared: twice the loss
        return base(m) + (5.0 if c < 4 else 0.0) - (0.0 if m == "lidar" else 20.0 * lv)
    iid = summary(models + ["win3_n"], {"trans": lv_t, "rot": lv_r, "drift": [0.0, 0.25]}, iid_value)
    sh = summary(models, {"trans_shared": lv_t, "rot_shared": lv_r}, shared_value)
    assert mt.shared_columns(sh, iid) == [("trans", "trans_shared", 0.1), ("trans", "trans_shared", 0.5), ("rot", "rot_shared", 0.1), ("rot", "rot_shared", 1.0)]
    val, rows = mt.shared_values(sh, iid)
    assert [(b[:9], l, n) for b, l, n in rows] == [("", "LiDAR-only", ["lidar"]), ("", "ImageVote", ["imagevote"]),
                                                   ("Delivered", "Late fusion", ["late_nat_b1", "late_n_s1"]), ("Delivered", "Feature concat.", ["concat_n"]),
                                                   ("Pose-nois", "Late fusion", ["late_a"])]
    all_mean = lambda b: b + 5.0 * 4 / 11              # mean over 11 classes of which 4 are 5 points higher
    assert abs(val[("Late fusion", rows[2][0], "trans_shared", 0.5, "mean_all")] - (all_mean(81.0) - 10.0)) < 1e-9      # seeds 80 and 82
    assert abs(val[("Late fusion", rows[2][0], "trans", 0.5, "mean_electrical")] - (86.0 - 5.0)) < 1e-9
    t = mt.shared_table(sh, iid)
    lines = [l.strip() for l in t.splitlines()]
    assert "\\begin{tabular}{@{}llrlrlrlrl@{\\hspace{14pt}}lrlrlrlrl@{}}" in t and "\\label{tab:shared}" in t
    assert lines.count("\\cmidrule(lr){2-10} \\cmidrule(lr){11-19}") == 1
    assert "Model & Clean & ind. & shared & ind. & shared & ind. & shared & ind. & shared & Clean & ind. & shared & ind. & shared & ind. & shared & ind. & shared \\\\" in lines
    # clean | 10 cm ind, shared | 50 cm | 0.1 deg | 1 deg ; all classes, then electrical classes
    a = all_mean(81.0)
    want = [a, a - 1, a - 2, a - 5, a - 10, a - 1, a - 2, a - 10, a - 20] + [86.0, 85.0, 84.0, 81.0, 76.0, 85.0, 84.0, 76.0, 66.0]
    assert "Late fusion & " + " & ".join(f"{v:.1f}" for v in want) + " \\\\" in lines
    lid = all_mean(70.0)
    assert "LiDAR-only & " + " & ".join([f"{lid:.1f}"] * 9 + ["75.0"] * 9) + " \\\\" in lines
    assert sum(l.startswith("Late fusion &") for l in lines) == 2 and not any(l.startswith("Window") for l in lines)   # win3_n: not in the shared run
    assert "\\multicolumn{19}{@{}l}{\\emph{Delivered poses in training}} \\\\" in lines and "mean over 1 or 2 training seeds" in t
    assert "Mean over 10 draws per level; exact re-association" in t
    # the two runs must hold the same heads on the same points: level 0 is one and the same evaluation
    other = summary(models, {"trans_shared": lv_t, "rot_shared": lv_r}, lambda m, sw, lv, c: shared_value(m, sw, lv, c) + (0.2 if m == "late_a" else 0.0))
    with pytest.raises(ValueError, match="late_a"):
        mt.shared_table(other, iid)
    fast = summary(models, {"trans_shared": lv_t, "rot_shared": lv_r}, shared_value); fast["meta"]["mode"] = "fast"
    with pytest.raises(ValueError, match="mode or draws"):
        mt.shared_table(fast, iid)
    # the repository files: run shared_v1 against exact_v1
    res = os.path.join(ROOT, "results")
    sh_real, iid_real = (json.load(open(os.path.join(res, f"protocol_{r}_summary.json"))) for r in (mt.SHARED_RUN, mt.EXACT_RUN))
    real = [l.strip() for l in mt.shared_table(sh_real, iid_real).splitlines()]
    assert ("Late fusion & 81.8 & 81.7 & 81.3 & 80.2 & 78.3 & 81.6 & 81.3 & 76.8 & 74.6 & 98.0 & 97.9 & 97.1 & 96.6 & 93.2 & 97.9 & 97.8 & 95.9 & 92.8 \\\\") in real
    assert ("Window $5{\\times}5$ & 79.2 & 79.1 & 78.9 & 76.3 & 72.0 & 79.1 & 78.9 & 72.2 & 65.3 & 98.0 & 98.0 & 97.5 & 91.8 & 81.8 & 98.0 & 98.0 & 87.0 & 71.5 \\\\") in real
    assert ("Late fusion & 82.0 & 82.0 & 81.9 & 81.3 & 80.3 & 81.9 & 81.7 & 78.7 & 77.4 & 98.0 & 97.9 & 97.9 & 97.8 & 97.7 & 97.9 & 97.9 & 97.8 & 97.7 \\\\") in real
    assert ("ImageVote & 65.5 & 63.3 & 59.9 & 51.3 & 49.9 & 65.1 & 62.1 & 46.3 & 43.4 & 63.7 & 58.0 & 49.2 & 27.1 & 27.4 & 62.9 & 55.6 & 22.0 & 20.2 \\\\") in real
    assert sh_real["meta"]["mode"] == iid_real["meta"]["mode"] == "exact" and sh_real["meta"]["draws"] == 10 and len(sh_real["meta"]["models"]) == 26
    # the independent half of the table is the fix / robustness tables' own source: same numbers as there
    fix = open(os.path.join(ROOT, "paper", "tables", "fix_val.tex"), encoding="utf-8").read()
    assert "Window $5{\\times}5$ & 3 & 98.0{\\scriptsize\\,$\\pm$\\,0.1} & 97.1{\\scriptsize\\,$\\pm$\\,0.4} & 91.8{" in fix


def test_shared_report_of_the_paper_numbers():
    """scripts/paper_numbers.py shared_report: the statements of the paragraph "Errors shared by all images"."""
    spec = importlib.util.spec_from_file_location("paper_numbers", os.path.join(ROOT, "scripts", "paper_numbers.py"))
    pn = importlib.util.module_from_spec(spec); spec.loader.exec_module(pn)
    lines = []
    pn.shared_report(os.path.join(ROOT, "results"), out=lines.append)
    text = "\n".join(lines)
    assert "runs shared_v1 / exact_v1 (exact, 10 draws; 12 heads trained with the delivered poses, 12 with pose noise)" in text
    assert "delivered poses: largest difference between the two runs 0.000" in text
    assert "deliv. shared minus independent, electrical, 50 cm: mean -7.28  range -10.32 .. -2.52  lower in 12 of 12" in text
    assert "deliv. shared minus independent, mIoU, 0.1 deg: mean -0.23  range -0.32 .. -0.14  lower in 12 of 12" in text
    assert ("deliv. shared reference levels: largest loss of mIoU 1.14 (concat_n_s1, trans_shared 0.1), of the electrical mean 2.75 "
            "(concat_n_s1, trans_shared 0.1), of one class 7.26 (concat_n_s1, structural, trans_shared 0.1)") in text
    assert "aug. shared reference levels: largest loss of mIoU 0.34 (late_a_s2, rot_shared 0.1), of the electrical mean 0.12" in text
    assert "deliv. Feature concat.   loss at shared 10 cm: mean_all 0.98  mean_electrical 2.34  conductor 1.51  structural 5.96  insulator 1.71" in text
    assert "deliv. Late fusion       loss of the electrical mean, shared (independent): 50 cm 4.80 (1.43)  1 deg 5.24 (2.12)" in text
    assert "deliv. Window 3x3        loss of the electrical mean, shared (independent): 50 cm 21.12 (11.46)  1 deg 32.31 (18.17)   mIoU minus LiDAR-only, shared: 50 cm +0.72  1 deg -6.53" in text
    assert "augmented heads (12): largest loss of the electrical mean over all levels of the shared sweeps: 0.58 (win5_a_s2, rot_shared 1)" in text
    assert "augmented head types, mIoU at the largest levels: independent minus shared 0.57 .. 1.36" in text
    assert "imagevote at 10 cm: loss of the electrical mean shared 14.51 / independent 5.67; of mIoU 5.55 / 2.11" in text
    assert "seed pairs, augmentation minus delivered, electrical, shared 50 cm: n=12 mean +14.68 range +2.92 .. +21.95 positive 12/12 p 0.0002" in text
    assert "seed pairs, augmentation minus delivered, electrical, shared 1 deg: n=12 mean +22.31 range +2.80 .. +33.54 positive 12/12 p 0.0002" in text
    assert "deliv. Window 5x5 minus concatenation, electrical, 50 cm: shared mean +1.77 range -0.09 .. +4.05 positive 2/3   independent mean +5.51 positive 3/3" in text
    assert "deliv. Window 3x3 minus concatenation, electrical, 1 deg: shared mean -5.49 range -8.60 .. -2.32 positive 0/3" in text
    assert "zones, augmentation minus delivered (12 head pairs per zone), electrical, shared 1 deg: mean +23.80 range +18.42 .. +34.19 positive 6/6" in text
    assert "zones, deliv. heads, shared minus independent, electrical, 50 cm: mean -8.03 range -11.43 .. -1.98 lower in 6 of 6" in text
    assert "deliv. std over the draws of the electrical mean: shared up to 8.34 (win3_n_s2, trans_shared 0.5)" in text
    assert "aug. std over the draws of the electrical mean: shared up to 0.14 (win5_a_s2, trans_shared 0.5)" in text

