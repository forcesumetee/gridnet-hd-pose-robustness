"""The paper tables are generated from result files: the generator must merge protocol runs, average over the seeds
that exist and leave out what is missing (pure Python: runs on the torch-less Mac venv too)."""
import importlib.util
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load_make_tables():
    spec = importlib.util.spec_from_file_location("make_tables", os.path.join(ROOT, "scripts", "make_tables.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def summary(models, base):
    """A protocol summary with flat curves: IoU = base[model] - 10 * level (in %), electrical 5 points higher."""
    sweeps = {"trans": [0.0, 0.02, 0.05, 0.1, 0.25, 0.5], "rot": [0.0, 0.05, 0.1, 0.25, 0.5, 1.0], "drift": [0.0, 0.1, 0.25, 0.5]}
    curves = []
    for m in models:
        for sw, levels in sweeps.items():
            for lv in levels:
                for cls, off in (("mean_all", 0.0), ("mean_electrical", 5.0)):
                    curves.append({"model": m, "sweep": sw, "level": lv, "class": cls,
                                   "iou_mean": (base[m] + off - 10 * lv * (m not in ("lidar",))) / 100, "iou_std": 0.0, "draws": 1})
    return {"meta": {"models": list(models), "sweeps": sweeps, "draws": 10, "mode": "fast", "version": 2,
                     "reference": {"trans": 0.1, "rot": 0.1, "drift": 0.25}}, "curves": curves, "scores": []}


def test_fix_and_ablation_tables_merge_runs_and_seeds():
    mt = load_make_tables()
    a = summary(["lidar", "imagevote", "late_nat_b1", "late_n_s1", "late_n_s2", "concat_n", "late_a", "late_a_s1", "concat_a", "win3_a"],
                {"lidar": 70.0, "imagevote": 60.0, "late_nat_b1": 80.0, "late_n_s1": 82.0, "late_n_s2": 84.0, "concat_n": 78.0,
                 "late_a": 81.0, "late_a_s1": 83.0, "concat_a": 79.0, "win3_a": 79.5})
    b = summary(["lidar", "late_a", "late_a_s2", "concat_a10"], {"lidar": 71.0, "late_a": 99.0, "late_a_s2": 85.0, "concat_a10": 77.0})
    val, owner = mt.merged_curves([("lab2", a), ("lab3", b)])
    assert owner["late_a"] == "lab2" and owner["late_a_s2"] == "lab3"           # the first run wins on a shared name
    assert abs(val[("late_a", "trans", 0.0, "mean_all")] - 81.0) < 1e-9 and abs(val[("lidar", "rot", 1.0, "mean_all")] - 70.0) < 1e-9
    fix = mt.fix_table([("lab2", a), ("lab3", b)])
    rows = [l.strip() for l in fix.splitlines()]
    late = [l for l in rows if l.startswith("Late fusion &")]
    assert len(late) == 4                                                       # two blocks x (delivered poses, augmentation)
    assert late[0].startswith("Late fusion & 3 & 82.0{\\scriptsize\\,$\\pm$\\,2.0}")       # 80, 82, 84
    assert late[1].startswith("Late fusion & 3 & 83.0{\\scriptsize\\,$\\pm$\\,2.0}")       # 81, 83 (lab2) + 85 (lab3)
    assert late[2].startswith("Late fusion & 3 & 87.0")                         # electrical block: 5 points higher
    concat = [l for l in rows if l.startswith("Feature concat. &")]
    assert concat[0].startswith("Feature concat. & 1 & 78.0 &") and "pm" not in concat[0]   # one seed: no spread
    assert not any(l.startswith("Window $3{\\times}3$ & 0") for l in rows)      # a head type without any run has no row
    assert sum(l.startswith("Window $3{\\times}3$ &") for l in rows) == 2       # win3_a only: the augmentation rows
    assert "by more than 1.0 mIoU" in fix                                       # largest change up to 10 cm / 0.1 deg: 10 x 0.1
    assert "\\texttt{lab2}, \\texttt{lab3}" in fix and "LiDAR-only & -- & 70.0" in fix
    abl = mt.ablation_table([("lab2", a), ("lab3", b)])
    assert "$\\le$\\,10\\,cm, 0.2$^\\circ$ & 77.0 & 82.0 & 72.0 & 77.0 & 67.0 & 72.0" in abl      # concat_a10 from the second run
    assert "\\quad probabilities + features" not in abl and "\\quad features + probabilities" not in abl   # heads that were not run
    assert "\\quad 5 &" not in abl                                               # have no row
    assert "\\quad features + features & 78.0" in abl                           # concat_n was run: the first row of the input block
    assert abl.count("\\midrule") == 1 + abl.count("\\emph{") - 1               # one rule per heading after the first
    only = mt.ablation_table([("x", summary(["lidar", "late_a"], {"lidar": 70.0, "late_a": 81.0}))])
    assert only.count("\\emph{") == 1 and "Range of the training noise, late fusion" in only   # headings without rows are dropped


def test_tables_from_the_result_files_of_the_repo():
    mt = load_make_tables()
    res = os.path.join(ROOT, "results")
    summaries = [(run, json.load(open(os.path.join(res, f"protocol_{run}_summary.json")))) for run in mt.FIX_RUNS
                 if os.path.exists(os.path.join(res, f"protocol_{run}_summary.json"))]
    assert [r for r, _ in summaries] == ["lab2", "lab3"]
    fix = mt.fix_table(summaries)
    # heads lab 2 (2026-11-13): three seeds without augmentation; concat loses the electrical classes at 1 deg
    assert "Late fusion & 3 & 81.8{\\scriptsize\\,$\\pm$\\,0.1}" in fix and "Feature concat. & 3 & 98.0{\\scriptsize\\,$\\pm$\\,0.1}" in fix
    assert fix.count("\\begin{tabular}") == 1 and fix.count("\\\\") >= 20
    # heads lab 3: the third seed of the augmented heads and the 5x5 window trained with the delivered poses
    assert "Late fusion & 3 & 82.0{\\scriptsize\\,$\\pm$\\,0.0} & 81.8" in fix and "Feature concat. & 3 & 80.6{\\scriptsize\\,$\\pm$\\,1.0}" in fix
    assert "Window $5{\\times}5$ & 1 & 79.5 & 79.1 & 76.9 & 77.5 & 72.7" in fix and "Window $5{\\times}5$ & 1 & 98.1 & 97.5 & 92.6 & 96.0 & 88.0" in fix
    abl = mt.ablation_table(summaries)
    # update 51: the table is one column wide; the rows of the last block name the two inputs in short, LiDAR first
    assert "\\multicolumn{7}{@{}l}{\\emph{Inputs of the concatenation head, LiDAR + image}}" in abl
    assert "\\quad probabilities + features & 81.9 & 98.0 & 74.0 & 79.5 & 67.2 & 69.8" in abl
    assert "\\begin{table}[!t]" in abl and "\\begin{tabular}{@{}lrrrrrr@{}}" in abl and "what the concatenation head receives" in abl
    assert "$\\le$\\,10\\,cm, 0.2$^\\circ$ & 79.4 & 98.1 & 78.0 & 96.2 & 75.3 & 94.8" in abl          # concat_a10 (lab 3)
    assert "\\quad probabilities + features, pose noise & 82.0 & 98.0 & 81.1 & 97.8 & 78.7 & 97.8" in abl
    assert "rotation only ($\\le$\\,1$^\\circ$) & 81.2" in abl


def test_exact_run_replaces_the_fast_labs(tmp_path):
    mt = load_make_tables()
    base = {"lidar": 70.0, "late_a": 81.0}

    def put(run, shift=0.0):
        json.dump(summary(["lidar", "late_a"], {k: v + shift for k, v in base.items()}), open(tmp_path / f"protocol_{run}_summary.json", "w"))

    assert mt.fix_summaries(str(tmp_path)) == []
    put("lab3", 1.0)
    assert [r for r, _ in mt.fix_summaries(str(tmp_path))] == ["lab3"]
    put("lab2")
    assert [r for r, _ in mt.fix_summaries(str(tmp_path))] == ["lab2", "lab3"]             # the order of FIX_RUNS, not of the files
    put(mt.EXACT_RUN, 2.0)
    found = mt.fix_summaries(str(tmp_path))
    assert [r for r, _ in found] == [mt.EXACT_RUN]                                         # never mixed with the fast runs
    assert "Late fusion & 1 & 83.0 &" in mt.fix_table(found)


def test_views_table():
    mt = load_make_tables()
    models = ["imagevote", "late_nat_b1", "late_a", "concat_n"]
    by_v = {V: summary(models, {"imagevote": 50.0 + V, "late_nat_b1": 70.0 + V, "late_a": 80.0 + V, "concat_n": 60.0 + V}) for V in (8, 1, 4)}
    by_v[1]["curves"] = [c for c in by_v[1]["curves"] if c["model"] != "concat_n"]          # a head missing for one V has no row
    t = mt.views_table(by_v)
    rows = [l.strip() for l in t.splitlines()]
    assert "Model \\hfill $V$\\,= & 1 & 4 & 8 & 1 & 4 & 8 & 1 & 4 & 8 \\\\" in rows
    assert "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}" in t and "\\begin{tabular}{lrrrrrrrrr}" in t
    # clean | sigma_t 50 cm (-5) | sigma_r 1 deg (-10), each for V = 1, 4, 8
    assert "Late fusion + aug. & 81.0 & 84.0 & 88.0 & 76.0 & 79.0 & 83.0 & 71.0 & 74.0 & 78.0 \\\\" in rows
    assert "ImageVote & 56.0 & 59.0 & 63.0 & 51.0 & 54.0 & 58.0 & 46.0 & 49.0 & 53.0 \\\\" in rows     # electrical block: 5 points higher
    assert not any(l.startswith("Feature concat.") for l in rows) and sum(l.startswith("Late fusion &") for l in rows) == 2
    assert "trained with 8 views" in t and "mean over\n  10 draws" in t


def test_paper_numbers_report():
    spec = importlib.util.spec_from_file_location("paper_numbers", os.path.join(ROOT, "scripts", "paper_numbers.py"))
    pn = importlib.util.module_from_spec(spec); spec.loader.exec_module(pn)
    mt = load_make_tables()
    res = os.path.join(ROOT, "results")
    lines = []
    pn.report([(run, json.load(open(os.path.join(res, f"protocol_{run}_summary.json")))) for run in mt.FIX_RUNS], out=lines.append)
    text = "\n".join(lines)
    assert "== aug. Late fusion  n=3  ['late_a', 'late_a_s1', 'late_a_s2']" in text and "== deliv. Window 5x5  n=1  ['win5_n']" in text
    assert "largest |change| of mIoU up to the reference levels: 0.17 (late_n_s2, rot 0.1)" in text
    assert "augmented heads (10): largest loss of the electrical mean over all levels of all sweeps: 0.35" in text
    assert "aug. Feature concat.         mIoU clean 80.65  loss at 50 cm 0.81  loss at 1 deg 3.13  gain over LiDAR-only at 1 deg +7.66" in text


def test_cost_table_from_the_repo_file():
    mt = load_make_tables()
    cost = json.load(open(os.path.join(ROOT, "results", "compute_cost.json")))
    assert all("source" in r for r in cost["rows"] if "component" in r)          # every number names where it was measured
    t = mt.cost_table(cost)
    assert "\\stage Late fusion head & 7.0\\,k & -- \\\\" in t and "37.8\\,k / 38.8\\,k" in t and "12.7\\,M & 7.0 h" in t
    # update 51: one column of the page. The stage texts wrap in a paragraph column, ragged, later lines indented
    assert "\\begin{table}[!t]" in t and f"\\begin{{tabular}}{{@{{}}p{{{mt.COST_STAGE_WIDTH}}}rr@{{}}}}" in t
    assert "\\def\\stage{\\raggedright\\hangindent=1.8em\\hangafter=1\\hspace*{1em}}" in t
    body = t.split("\\midrule", 1)[1].split("\\bottomrule")[0].splitlines()
    rows = [l.strip() for l in body if "&" in l]
    assert len(rows) == sum("component" in r for r in cost["rows"]) and all(l.startswith("\\stage ") and l.count("&") == 2 for l in rows)
    assert t.count("\\midrule") == 1 + sum("group" in r for r in cost["rows"]) - 1 and "\\label{tab:cost}" in t
    assert mt._params(None) == "--" and mt._params(33852) == "33.9\\,k" and mt._params([1000, 2500000]) == "1.0\\,k / 2.5\\,M"


def test_exact_signed_rank_test():
    from gridnet.stats import paired_summary, signed_rank_exact
    assert signed_rank_exact([1, 2, 3]) == 0.125 and signed_rank_exact([0.3] * 9) == 2 ** -9          # all pairs agree: 2^-n
    assert signed_rank_exact([1, 2, 3], "less") == 1.0 and signed_rank_exact([1, 2, 3], "two-sided") == 0.25
    assert signed_rank_exact([1, -1, 2, -2], "two-sided") == 1.0 and signed_rank_exact([0, 0]) == 1.0   # zeros are dropped
    # reference values of scipy.stats.wilcoxon(d, alternative=..., method="exact")
    d = [0.9, -0.3, 1.7, 2.2, -0.1, 0.6]
    assert abs(signed_rank_exact(d) - 0.078125) < 1e-12 and abs(signed_rank_exact(d, "two-sided") - 0.15625) < 1e-12
    assert abs(signed_rank_exact(d, "less") - 0.953125) < 1e-12
    s = paired_summary([1.0, 3.0, -1.0], "two-sided")
    assert s["n"] == 3 and s["mean"] == 1.0 and s["positive"] == 2 and s["min"] == -1.0 and s["max"] == 3.0
    assert paired_summary([])["n"] == 0


def test_stats_table_pairs_heads_by_seed():
    mt = load_make_tables()
    models = ["lidar", "late_nat_b1", "late_n_s1", "late_a", "late_a_s1", "concat_n", "concat_a", "win3_n", "win3_a"]
    base = {"lidar": 70.0, "late_nat_b1": 80.0, "late_n_s1": 81.0, "late_a": 82.0, "late_a_s1": 85.0, "concat_n": 78.0, "concat_a": 79.5,
            "win3_n": 78.5, "win3_a": 79.0}
    rows = mt.stat_rows([("lab2", summary(models, base))])
    aug = dict(rows[0][2])["All classes, delivered poses"]
    assert rows[0][1] == "greater" and aug["n"] == 4 and aug["positive"] == 4 and abs(aug["mean"] - (2 + 4 + 1.5 + 0.5) / 4) < 1e-9
    assert abs(aug["p"] - 2 ** -4) < 1e-12                                       # late x2 seeds, concat, window3: all four agree
    win = dict(rows[1][2])["All classes, delivered poses"]
    assert rows[1][1] == "two-sided" and win["n"] == 2 and win["positive"] == 1   # +0.5 without, -0.5 with augmentation
    late = dict(rows[2][2])
    assert len(rows) == 3 and list(late.values())[0]["n"] == 1                   # late_nat_b1 vs concat_n only (seed 0)
    t = mt.stats_table([("lab2", summary(models, base))])
    assert "\\quad All classes, delivered poses & 4 & $+2.0$ & $+0.5$ to $+4.0$ & 4/4 & 0.062 \\\\" in t and "two-sided test" in t
    # the repository files: ten pairs for augmentation (three head types x three seeds + the 5x5 window with one seed)
    res = os.path.join(ROOT, "results")
    real = [(run, json.load(open(os.path.join(res, f"protocol_{run}_summary.json")))) for run in mt.FIX_RUNS]
    r = mt.stat_rows(real)
    el = dict(r[0][2])["Electrical classes, $\\sigma_t$\\,=\\,50\\,cm"]
    assert el["n"] == 10 and el["positive"] == 10 and round(el["mean"], 1) == 7.8 and abs(el["p"] - 2 ** -10) < 1e-12
    assert dict(r[1][2])["All classes, delivered poses"]["p"] > 0.4               # window vs concatenation: no difference


def test_paper_stats_zone_pairs():
    spec = importlib.util.spec_from_file_location("paper_stats", os.path.join(ROOT, "scripts", "paper_stats.py"))
    ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)
    names = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb", "rock_soil", "road", "water", "building"]

    def cm(acc):                       # two classes present (conductor, herb), `acc` of the conductor points right
        m = [[0.0] * 11 for _ in range(11)]
        m[1][1], m[1][6], m[6][6] = 100 * acc, 100 * (1 - acc), 1000.0
        return m

    runs = []
    for zi, zone in enumerate(["za", "zb", "zc"]):
        for model, acc in (("concat_n", 0.5 + 0.1 * zi), ("concat_a", 0.9)):
            for draw in (0, 1):
                runs.append({"model": model, "sweep": "trans", "level": 0.5, "draw": draw, "zone": zone, "coverage": 1.0, "cm": cm(acc)})
    res = {"meta": {"class_names": names}, "runs": runs}
    iou, row = ps.iou_per_class(cm(0.5))
    assert iou[1] == 0.5 and iou[0] is None and row[1] == 100
    val = ps.zone_values([res], "mean_electrical")
    assert abs(val[("concat_a", "trans", 0.5, "za")] - 90.0) < 1e-9 and abs(val[("concat_n", "trans", 0.5, "zc")] - 70.0) < 1e-9
    d = ps.zone_pairs([res], ["concat_n"], ["concat_a"], "trans", 0.5, "mean_electrical")
    assert [z for z, _ in d] == ["za", "zb", "zc"] and [round(x) for _, x in d] == [40, 30, 20]
    assert ps.zone_pairs([res], ["concat_n", "missing"], ["concat_a"], "trans", 0.5, "mean_electrical") == []   # incomplete group: no pair
    lines = []
    ps.report([("lab2", summary(["lidar", "concat_n", "concat_a"], {"lidar": 70.0, "concat_n": 78.0, "concat_a": 79.0}))], [res], out=lines.append)
    text = "\n".join(lines)
    assert "zones as pairs" in text and "n= 3  mean +30.00" in text and "p 0.1250 (greater)" in text
    # the per-zone file the statistics table is made from: every metric of every (model, sweep, level, zone)
    doc = ps.zone_doc([res])
    assert doc["zones"] == ["za", "zb", "zc"] and doc["metrics"] == ps.ZONE_METRICS and len(doc["rows"]) == 6
    assert ["concat_a", "trans", 0.5, "za", 94.505, 90.0, 94.505] in doc["rows"]          # conductor 90 %, herb 1000 / 1010
    mt = load_make_tables()
    zval = mt.zone_values_dict(json.loads(json.dumps(doc)))
    assert abs(zval[("concat_n", "trans", 0.5, "zc", "mean_electrical")] - 70.0) < 1e-9
    rows = dict(mt.zone_stat_rows(zval)[0][3])
    assert rows["Electrical classes, $\\sigma_t$\\,=\\,50\\,cm"]["n"] == 3 and round(rows["Electrical classes, $\\sigma_t$\\,=\\,50\\,cm"]["mean"]) == 30


def test_test_set_table_from_the_repo_file():
    mt = load_make_tables()
    data = json.load(open(os.path.join(ROOT, "results", "test_set.json")))
    t = mt.test_table(data)
    assert "\\quad ImageVote~\\cite{gridnethd} & 60.0\\,M & 69.11 & 86.89 \\\\" in t and "DITR, test-time augmentation~\\cite{gridnethd} & 46.7\\,M & 74.87 & 87.03" in t
    assert "read from it on 14 November 2026" in t and "\\label{tab:test}" in t
    ours = [r for r in data["rows"] if r.get("ours")]
    assert [r["ours"] for r in ours] == ["lidar", "fused"]
    for r in ours:                                             # a score that is not entered yet prints as a todo, never as a number
        line = next(l for l in t.splitlines() if r["method"] in l)
        assert (r["miou"] is None) == ("\\todo{--}" in line)
    filled = json.loads(json.dumps(data))
    for r in filled["rows"]:
        if r.get("ours") == "fused":
            r["miou"], r["oa"] = 77.123, 88.0
    t2 = mt.test_table(filled)
    assert "Late fusion with pose-noise augmentation, ours & 49.7\\,M & 77.12 & 88.00 \\\\" in t2
    # the published rows carry the per-class values of the dataset paper, and their mean is the paper's mIoU
    for r in data["rows"]:
        if r.get("per_class"):
            assert len(r["per_class"]) == len(data["class_order"]) == 11
            assert abs(sum(r["per_class"]) / 11 - r["miou"]) < 0.02, r["method"]
    # every row of the table is on the leaderboard snapshot with the same scores
    snap = {e[0]: (e[2], e[3]) for e in data["leaderboard_snapshot"]}
    for r in data["rows"]:
        if r.get("entry"):
            assert snap[r["entry"]] == (r["miou"], r["oa"]), r["entry"]



def test_clean_table_from_the_heads_labs():
    mt = load_make_tables()

    def e(m, water):
        return {"mIoU": m, "OA": m + 10, "per_class": {k: (water if k == "water" else m) for k, _ in mt.NAMES}, "per_zone": {}}

    val = {"lidar": e(70.0, 0.0), "imagevote": e(65.0, 40.0), "late_nat_b1": e(80.0, 30.0), "late_n_s1": e(82.0, 34.0),
           "late_n_s2": e(84.0, 38.0), "concat_n": e(79.0, 5.0), "late_a": e(83.0, 41.0)}
    t = mt.clean_bank_table(val)
    rows = [l.strip() for l in t.splitlines()]
    assert "Class & LiDAR-only & ImageVote & Late & Concat. & Late \\\\" in rows                 # head types without a head: no column
    assert "\\multicolumn{2}{c}{Delivered poses in training} & \\multicolumn{1}{c}{Pose noise}" in t
    assert "Water & 0.0 & 40.0 & 34.0 & 5.0 & \\textbf{41.0} \\\\" in rows                        # mean of the seeds: 30, 34, 38
    assert "mIoU & 70.0 & 65.0 & 82.0 & 79.0 & \\textbf{83.0} \\\\" in rows
    assert "\\quad std over seeds & -- & -- & 2.0 & -- & -- \\\\" in rows
    assert "OA & 80.0 & 75.0 & 92.0 & 89.0 & \\textbf{93.0} \\\\" in rows and "mean over 1 or 3 training seeds" in t
    # the repository files: three seeds per head type, and the same numbers as level 0 of the exact protocol run
    res = os.path.join(ROOT, "results")
    bank = mt.bank_val(res)
    real = mt.clean_bank_table(bank)
    assert "mIoU & 69.9 & 65.5 & 81.8 & 79.2 & 79.6 & 79.2 & \\textbf{82.0} \\\\" in real
    assert "\\quad std over seeds & -- & -- & 0.1 & 0.4 & 0.6 & 0.4 & 0.0 \\\\" in real
    assert "Water & 0.1 & \\textbf{42.7} & 38.7 & 4.8 & 9.2 & 5.6 & 41.3 \\\\" in real and "mean over 3 training seeds" in real
    val, _ = mt.merged_curves([(mt.EXACT_RUN, json.load(open(os.path.join(res, f"protocol_{mt.EXACT_RUN}_summary.json"))))])
    for _, names in mt.FIX_GROUPS:
        for n in names or []:
            assert abs(bank[n]["mIoU"] - val[(n, "trans", 0.0, "mean_all")]) < 1e-3, n
            assert abs(bank[n]["per_class"]["water"] - val[(n, "trans", 0.0, "water")]) < 1e-3, n


def test_robustness_table_rows_are_seed_means():
    mt = load_make_tables()
    s = summary(["lidar", "imagevote", "late_nat_b1", "late_n_s1", "concat_n"],
                {"lidar": 70.0, "imagevote": 60.0, "late_nat_b1": 80.0, "late_n_s1": 84.0, "concat_n": 78.0})
    s["meta"]["mode"] = "exact"
    t = mt.robustness_table(s, "exact_v1", [("lidar", "LiDAR-only"), (["late_nat_b1", "late_n_s1", "late_n_s2"], "Late fusion"),
                                            (["win3_n"], "Window $3{\\times}3$")])
    rows = [l.strip() for l in t.splitlines()]
    # clean | 10 cm | 50 cm | 0.1 deg | 1 deg | drift 25 | drift 50 ; a synthetic curve loses 10 points per unit of level
    assert "Late fusion & 82.0 & 81.0 & 77.0 & 81.0 & 72.0 & 79.5 & 77.0 \\\\" in rows                # the two seeds that exist
    assert "Late fusion & 87.0 & 86.0 & 82.0 & 86.0 & 77.0 & 84.5 & 82.0 \\\\" in rows                # electrical block
    assert "LiDAR-only & 70.0 & 70.0 & 70.0 & 70.0 & 70.0 & 70.0 & 70.0 \\\\" in rows
    assert not any(l.startswith("Window") for l in rows) and "mean over 2 training seeds" in t and "exact re-association" in t
    old = mt.robustness_table(s, "fast_v1", [("lidar", "LiDAR-only"), ("concat_n", "Feature concat.")])     # single models as before
    assert "Feature concat. & 78.0 & 77.0 & 73.0" in old and "training seeds" not in old
    real = mt.robustness_table(json.load(open(os.path.join(ROOT, "results", f"protocol_{mt.EXACT_RUN}_summary.json"))), mt.EXACT_RUN,
                               [("imagevote", "ImageVote"), (mt._group("Window $5{\\times}5$", "Delivered"), "Window $5{\\times}5$")])
    assert "ImageVote & 63.7 & 58.0 & 27.1 & 62.9 & 22.0 & 56.7 & 46.1 \\\\" in real                  # electrical classes, exact form
    assert "Window $5{\\times}5$ & 98.0 & 98.0 & 91.8 & 98.0 & 87.0 & 97.8 & 96.6 \\\\" in real


def test_zone_part_of_the_stats_table():
    mt = load_make_tables()
    zval = {}
    for zi, z in enumerate(["za", "zb", "zc", "zd"]):
        delta = 1.0 if zi % 2 else -1.0
        for m, v in (("concat_n", 80.0 - 5 * zi), ("concat_a", 95.0), ("win3_n", 80.0 - 5 * zi + delta), ("win3_a", 95.0)):
            for cls in ("mean_electrical", "mean_all", "mean_wo_water"):
                for sw, lv in (("trans", 0.5), ("rot", 1.0), ("trans", 0.0)):
                    zval[(m, sw, lv, z, cls)] = v
    blocks = mt.zone_stat_rows(zval)
    title, alt, k, rows = blocks[0]
    d = dict(rows)["Electrical classes, $\\sigma_t$\\,=\\,50\\,cm"]
    # per zone: mean of the two augmented heads minus mean of their twins = 15.5, 19.5, 25.5, 29.5
    assert alt == "greater" and k == 2 and d["n"] == 4 and d["positive"] == 4 and abs(d["mean"] - 22.5) < 1e-9 and abs(d["p"] - 2 ** -4) < 1e-12
    win = dict(blocks[1][3])["All classes, delivered poses"]
    assert blocks[1][1] == "two-sided" and win["positive"] == 2 and abs(win["mean"]) < 1e-9 and win["p"] == 1.0   # -0.5, +0.5, -0.5, +0.5
    assert len(blocks) == 2                                                    # no late or 5x5 heads: those comparisons are left out
    del zval[("win3_a", "trans", 0.5, "zd", "mean_electrical")]               # a zone without one of the heads gives no pair
    assert dict(mt.zone_stat_rows(zval)[0][3])["Electrical classes, $\\sigma_t$\\,=\\,50\\,cm"]["n"] == 3
    models = ["lidar", "concat_n", "concat_a", "win3_n", "win3_a"]
    summ = [("exact_v1", summary(models, {"lidar": 70.0, "concat_n": 78.0, "concat_a": 79.0, "win3_n": 78.5, "win3_a": 79.5}))]
    t = mt.stats_table(summ, zval)
    # update 51: seeds and zones side by side, one row per quantity (the table was twice as tall with one part below the other)
    assert "\\multicolumn{5}{c}{Training seeds as pairs} & \\multicolumn{5}{c}{Validation zones as pairs}" in t
    assert "\\begin{tabular}{@{}lrrcrr@{\\hspace{14pt}}rrcrr@{}}" in t and "Left: training seeds as pairs." in t
    assert "Right: a pair is a validation zone" in t and "as many as there are seed pairs on the left" in t
    assert t.count("\\quad Electrical classes, $\\sigma_t$\\,=\\,50\\,cm") == 2      # one per comparison, both halves in the row
    both = [l for l in t.splitlines() if l.strip().startswith("\\quad")]
    assert both and all(l.count("&") == 10 for l in both)
    heads = [l for l in t.splitlines() if "\\multicolumn{11}{@{}l}" in l]
    assert len(heads) == 2 and heads[0].rstrip().endswith("; one-sided test}} \\\\") and heads[1].rstrip().endswith("; two-sided test}} \\\\")
    # zd: the four zones give 4 of 4 for augmentation; a zone value that is missing prints as -- on its side only
    assert "& 4 & $+22.5$ & $+15.5$ to $+29.5$ & 4/4 & 0.062 \\\\" in t
    plain = mt.stats_table(summ)                                               # without zone values: the table of the seeds alone
    assert "Validation zones" not in plain and "Training seeds as pairs" not in plain and "\\begin{tabular}{lrrcrr}" in plain
    doc = {"metrics": ["mean_all", "mean_electrical"], "rows": [["m", "trans", 0.5, "za", 71.5, None]]}
    assert mt.zone_values_dict(doc) == {("m", "trans", 0.5, "za", "mean_all"): 71.5}
    # the repository file of the exact run: twelve head pairs, six zones
    real = mt.zone_values_dict(json.load(open(os.path.join(ROOT, "results", f"protocol_{mt.EXACT_RUN}_zones.json"))))
    aug = mt.zone_stat_rows(real)[0]
    el = dict(aug[3])["Electrical classes, $\\sigma_r$\\,=\\,1$^\\circ$"]
    assert aug[2] == 12 and el["n"] == 6 and el["positive"] == 6 and round(el["mean"], 1) == 13.6 and abs(el["p"] - 2 ** -6) < 1e-12
    # the caption of the side-by-side table says "as many head pairs as there are seed pairs on the left": true for every block
    seed_n = {title: {st["n"] for _, st in rows} for title, _, rows in
              mt.stat_rows([(mt.EXACT_RUN, json.load(open(os.path.join(ROOT, "results", f"protocol_{mt.EXACT_RUN}_summary.json"))))])}
    zone_blocks = mt.zone_stat_rows(real)
    assert [t for t, _, _, _ in zone_blocks] == list(seed_n) and len(zone_blocks) == 4
    for title, _, k, _ in zone_blocks:
        assert seed_n[title] == {k}, (title, k, seed_n[title])
    seeds = mt.stat_rows([(mt.EXACT_RUN, json.load(open(os.path.join(ROOT, "results", f"protocol_{mt.EXACT_RUN}_summary.json"))))])
    el = dict(seeds[0][2])["Electrical classes, $\\sigma_t$\\,=\\,50\\,cm"]
    assert el["n"] == 12 and el["positive"] == 12 and round(el["mean"], 1) == 7.6 and abs(el["p"] - 2 ** -12) < 1e-12
    k5 = dict(seeds[2][2])["Electrical classes, $\\sigma_t$\\,=\\,50\\,cm"]
    assert seeds[2][0].startswith("Window $5{\\times}5$ minus feature concatenation") and k5["n"] == 3 and k5["positive"] == 3


def test_displacement_table():
    mt = load_make_tables()
    s = summary(["lidar"], {"lidar": 70.0})
    s["meta"]["drift_deg"] = {"0.25": 0.25}
    row = lambda sw, lv, a, b, c: {"sweep": sw, "level": lv, "n": 60, "shift_px": a, "shift_px_p90": b, "shift_px_electrical": c, "coverage": 0.96}
    s["displacement"] = [row("trans", 0.0, 0.0, 0.0, 0.0), row("trans", 0.1, 6.44, 13.0, 10.26), row("rot", 1.0, 82.8, 156.6, 79.6),
                         row("drift", 0.25, 6.6, 16.4, 9.7)]
    t = mt.displacement_table(s, "exact_v1")
    assert "\\quad 10\\,cm & 6.4 & 13.0 & 10.3 \\\\" in t and "\\quad 1$^\\circ$ & 82.8 & 156.6 & 79.6 \\\\" in t
    assert "\\quad 25\\,cm, 0.25$^\\circ$ & 6.6 & 16.4 & 9.7 \\\\" in t and t.count("\\quad") == 3 and "\\label{tab:displacement}" in t


def test_paper_numbers_benchmark_statements():
    """The numbers that Sections 6.3-6.5 and the discussion quote outside the tables, from the files of the exact run."""
    spec = importlib.util.spec_from_file_location("paper_numbers", os.path.join(ROOT, "scripts", "paper_numbers.py"))
    pn = importlib.util.module_from_spec(spec); spec.loader.exec_module(pn)
    mt = load_make_tables()
    res = os.path.join(ROOT, "results")
    summaries = mt.fix_summaries(res)
    assert [r for r, _ in summaries] == [mt.EXACT_RUN] and summaries[0][1]["meta"]["mode"] == "exact"
    lines = []
    pn.report(summaries, out=lines.append)
    pn.benchmark_report(summaries, res, out=lines.append)
    text = "\n".join(lines)
    assert "== deliv. Window 5x5  n=3  ['win5_n', 'win5_n_s1', 'win5_n_s2']" in text
    assert "augmented heads (12): largest loss of the electrical mean over all levels of all sweeps: 0.36 (concat_a_s2, rot 1)" in text
    assert "deliv. Window 5x5            mIoU clean 79.19  loss at 50 cm 2.91  loss at 1 deg 6.97  gain over LiDAR-only at 1 deg +2.36" in text
    assert "change at 10 cm: mean_a -2.1 mean_e -5.7 pylon -1.9 conduc -6.8 struct -10.7" in text          # the vote baseline, exact form
    assert "trans 0.1: largest loss of mIoU 0.11 (concat_n_s1)" in text and "rot 0.1: largest loss of mIoU 0.16 (late_n_s2)" in text
    assert "drift 0.25: largest loss of mIoU 0.36 (concat_n_s1), of the electrical mean 0.57 (concat_n_s1), of one class 1.53 (win3_n_s1, structural)" in text
    assert "50 cm  conductor points predicted as high_veg 1.9 %, low_veg 6.3 %, herb 0.8 %, any other class 9.5 %" in text
    assert "deliv. late: at or above the better input in 10 of 11 classes; below it: {'water': -4.0}" in text
    assert "deliv. Feature concat.  water val 6.5 / 7.0 / 1.0   training zones 34.2 / 37.7 / 10.2" in text
    assert "without image evidence: largest shortfall of an electrical class against LiDAR-only 0.73 (win3_n, pylon)" in text
    assert "loss of the electrical mean at 1 deg: r = +0.33" in text and "win5_a_s2 29.9" in text
    assert "fusion heads, mean_all: largest |difference| 0.04" in text and "fusion heads, mean_electrical: largest |difference| 0.13 (concat_lp_n, trans 0.5)" in text
    assert "imagevote, mean_electrical, trans: largest difference +1.58 (level 0.1)" in text


def test_paper_numbers_motivating_experiment_and_qualitative_figure():
    """Sec. 6.1 (label transfer over the 852 images of the validation zones) and the paragraph on the qualitative figure."""
    spec = importlib.util.spec_from_file_location("paper_numbers", os.path.join(ROOT, "scripts", "paper_numbers.py"))
    pn = importlib.util.module_from_spec(spec); spec.loader.exec_module(pn)
    res = os.path.join(ROOT, "results")
    lines = []
    pn.motivating_report(res, out=lines.append)
    pn.qualitative_report(res, out=lines.append)
    text = "\n".join(lines)
    assert "852 images, error model protocol, 1 draw per image and level" in text
    assert "images per zone: t1z5b 133, t1z8 103, t3z4 153, t4z1 173, t5b1 136, t5b5 154; point-pixel pairs per image 3,279,609" in text
    assert "delivered poses  all 95.6  pylon 93.9  conductor 92.0  structural 91.7  insulator 84.6" in text
    assert "sigma_t 10 cm    all 94.8  pylon 87.7  conductor 70.4  structural 55.8  insulator 75.3" in text
    assert "sigma_r 0.1 deg  all 94.7  pylon 88.2  conductor 70.6  structural 65.3  insulator 76.4" in text
    assert "sigma_t 50 cm    all 91.7  pylon 68.6  conductor 25.9  structural 15.1  insulator 30.0" in text
    assert "sigma_r 1 deg    all 87.9  pylon 56.6  conductor 15.8  structural 10.3  insulator 17.2" in text
    assert "structural at sigma_t 10 cm: quartiles over the images 19.3 .. 82.0; zones 49.8 (t1z5b) .. 72.1 (t3z4)" in text
    assert "thin classes at 50 cm and 1 deg: 10.3 .. 30.0" in text
    assert "== qualitative figure: 12 pylons in 6 zones" in text
    assert "concat_n     delivered 0.2 .. 5.6 (0.4)   10 cm 0.2 .. 5.9 (0.4)   50 cm 0.4 .. 15.8 (5.1)   1 deg 0.5 .. 21.7 (7.0)" in text
    assert "concat_a     delivered 0.2 .. 2.8 (0.5)   10 cm 0.2 .. 2.8 (0.5)   50 cm 0.2 .. 2.8 (0.6)   1 deg 0.2 .. 3.4 (0.7)" in text
    assert "fusion heads at 10 cm: largest |change| against the delivered poses 0.25" in text
    assert "concat_a: largest increase at 1 deg over the delivered poses +1.20" in text
    # the two rows of the figure are read from the crop files themselves, which hold points of the dataset and are for
    # that reason not in the public repository (scripts/export_public.py); everything above comes from crops.json
    if not all(os.path.exists(os.path.join(res, "qualitative", f)) for f in ("t1z8_crop0.npz", "t4z1_crop0.npz")):
        return
    assert "figure row t1z8_crop0 (rank 12 of 12 at 1 deg): lidar 0.6, concat_n delivered 0.9 / 1 deg 21.7, concat_a 1 deg 0.7" in text
    assert "figure row t4z1_crop0 (rank 6 of 12 at 1 deg): lidar 0.8, concat_n delivered 0.4 / 1 deg 7.0, concat_a 1 deg 0.7" in text
    assert "right class for pylon 98.1, conductor 39.4, structural 29.0, insulator 59.0; conductor -> low_veg 57.2 %, structural -> low_veg 69.9 %" in text
    assert "conductor -> herb 18.6 %" in text
