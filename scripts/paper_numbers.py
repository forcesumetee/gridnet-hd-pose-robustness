"""Print the numbers that the text of Section 6 quotes, from the same result files as the tables (pure Python).

  python scripts/paper_numbers.py            # the runs behind the fix / ablation tables (exact run once it exists)
  python scripts/paper_numbers.py lab2 lab3  # explicit protocol runs (results/protocol_<run>_summary.json)

For every head type of Table "fix" (seeds merged): per-class IoU, mean +- std over seeds, with the delivered poses, at
the reference levels and at the largest levels; then the derived statements of the text (largest change up to the
reference levels, largest loss of the electrical classes of the augmented heads over all levels, ...). The text of
the paper is checked against this printout whenever the result files change.
With the default runs the benchmark numbers of Sections 6.3-6.4 and of the discussion follow (benchmark_report): the
vote baseline per class, the largest changes at the reference levels of all three error models, per-class losses of
the head types, where the conductor points of the concatenation head go (results/protocol_<run>_confusion.json), the
modality ablations and the water class of the heads labs (results/bank_*.json), the fast against the exact form.
Then the run with shared pose errors against the run with independent ones (shared_report: results/protocol_shared_v1_*
next to results/protocol_exact_v1_*), i.e. every number of the paragraph "Errors shared by all images".
"""
import importlib.util
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_spec = importlib.util.spec_from_file_location("make_tables", os.path.join(ROOT, "scripts", "make_tables.py"))
mt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mt)

CLASSES = ["mean_all", "mean_electrical", "pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb",
           "rock_soil", "road", "water", "building"]
COLUMNS = [("trans", 0.0, "clean"), ("trans", 0.1, "t 10cm"), ("trans", 0.5, "t 50cm"), ("rot", 0.1, "r 0.1"),
           ("rot", 1.0, "r 1.0"), ("drift", 0.25, "d 25cm"), ("drift", 0.5, "d 50cm")]


def groups(owner):
    """[(key, label, [names that exist])] for the head types of the fix table: late_n, late_a, concat_n, ..."""
    out, aug = [], False
    for label, names in mt.FIX_GROUPS:
        if names is None:
            aug = "augmentation" in label
            continue
        have = [n for n in names if n in owner]
        if have:
            out.append((names[0], ("aug. " if aug else "deliv. ") + label.replace("$", "").replace("{\\times}", "x"), have))
    return out


def stat(val, names, sw, lv, cls):
    xs = [val[(n, sw, lv, cls)] for n in names if (n, sw, lv, cls) in val]
    return mt._mean_std(xs) if xs else (None, None)


def cell(m, sd):
    return "     --     " if m is None else (f"{m:6.1f}      " if sd is None else f"{m:6.1f}±{sd:4.1f} ")


def report(summaries, out=print):
    val, owner = mt.merged_curves(summaries)
    meta = summaries[0][1]["meta"]
    out(f"runs {[r for r, _ in summaries]}  mode {meta['mode']}  draws {meta['draws']}  models {len(owner)}")
    gs = groups(owner)
    for _, label, names in gs:
        out(""); out(f"== {label}  n={len(names)}  {names}")
        out(f"{'class':16s}" + "".join(f"{c[2]:>12s}" for c in COLUMNS))
        for cls in CLASSES:
            out(f"{cls:16s}" + "".join(cell(*stat(val, names, sw, lv, cls)) for sw, lv, _ in COLUMNS))
    out(""); out("== derived statements")
    for key in ("lidar", "imagevote"):
        if key in owner:
            out(f"{key:10s} " + "  ".join(f"{c[2]} {val[(key, c[0], c[1], 'mean_all')]:.1f}/{val[(key, c[0], c[1], 'mean_electrical')]:.1f}" for c in COLUMNS))
    small = [(sw, float(lv)) for sw in ("trans", "rot") for lv in meta["sweeps"][sw] if 0 < float(lv) <= meta["reference"][sw]]
    heads = [n for _, _, names in gs for n in names]
    worst = max((abs(val[(n, sw, 0.0, "mean_all")] - val[(n, sw, lv, "mean_all")]), n, sw, lv) for n in heads for sw, lv in small)
    out(f"largest |change| of mIoU up to the reference levels: {worst[0]:.2f} ({worst[1]}, {worst[2]} {worst[3]:g})")
    worst = max((abs(val[(n, sw, 0.0, c)] - val[(n, sw, lv, c)]), n, sw, lv, c) for n in heads for sw, lv in small
                for c in ("pylon", "conductor", "structural", "insulator"))
    out(f"largest |change| of an electrical class up to the reference levels: {worst[0]:.2f} ({worst[1]}, {worst[2]} {worst[3]:g}, {worst[4]})")
    aug = [n for _, label, names in gs if label.startswith("aug.") for n in names]
    if aug:
        levels = [(sw, float(lv)) for sw in meta["sweeps"] for lv in meta["sweeps"][sw] if float(lv) > 0]
        worst = max((val[(n, sw, 0.0, "mean_electrical")] - val[(n, sw, lv, "mean_electrical")], n, sw, lv) for n in aug for sw, lv in levels)
        out(f"augmented heads ({len(aug)}): largest loss of the electrical mean over all levels of all sweeps: {worst[0]:.2f} ({worst[1]}, {worst[2]} {worst[3]:g})")
        for sw, lv, name in (("trans", 0.5, "50 cm"), ("rot", 1.0, "1 deg"), ("drift", 0.5, "drift 50 cm")):
            el = [val[(n, sw, lv, "mean_electrical")] for n in aug]; st = [val[(n, sw, lv, "structural")] for n in aug]
            out(f"  at {name}: electrical mean {min(el):.2f} .. {max(el):.2f}, structural cable {min(st):.2f} .. {max(st):.2f}")
        for cls in ("mean_all", "mean_electrical", "structural"):
            sd = max((stat(val, names, sw, lv, cls)[1] or 0.0, label, sw, lv) for _, label, names in gs if label.startswith("aug.") and len(names) > 1
                     for sw, lv, _ in COLUMNS) if any(label.startswith("aug.") and len(names) > 1 for _, label, names in gs) else None
            if sd:
                out(f"  largest std over seeds of an augmented head type, {cls}: {sd[0]:.2f} ({sd[1]}, {sd[2]} {sd[3]:g})")
    if "lidar" in owner:
        for _, label, names in gs:
            m1 = stat(val, names, "rot", 1.0, "mean_all")[0]; m5 = stat(val, names, "trans", 0.5, "mean_all")[0]
            c0 = stat(val, names, "trans", 0.0, "mean_all")[0]
            out(f"{label:28s} mIoU clean {c0:.2f}  loss at 50 cm {c0 - m5:.2f}  loss at 1 deg {c0 - m1:.2f}  "
                f"gain over LiDAR-only at 1 deg {m1 - val[('lidar', 'rot', 1.0, 'mean_all')]:+.2f}")
    return val, owner


def _pearson(xs, ys):
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5; sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def benchmark_report(summaries, res, out=print):
    """Numbers of Sections 6.3-6.4 and of the discussion that are not in the tables. Needs the heads of FIX_GROUPS."""
    val, owner = mt.merged_curves(summaries)
    run, meta = summaries[0][0], summaries[0][1]["meta"]
    thin, cover = ["pylon", "conductor", "structural", "insulator"], ["road", "water", "building", "rock_soil", "herb"]
    deliv = [(lab, [n for n in mt._group(lab, "Delivered") if n in owner]) for lab, _, _ in mt.CLEAN_HEADS[:4]]
    aug = [(lab, [n for n in mt._group(lab, "Pose-noise") if n in owner]) for lab, _, _ in mt.CLEAN_HEADS[:4]]
    plain = lambda lab: lab.replace("$", "").replace("{\\times}", "x")
    g = lambda names, sw, lv, cls: sum(val[(n, sw, lv, cls)] for n in names) / len(names)
    out(""); out("== benchmark: vote baseline per class (IoU)")
    for sw, lv, name in COLUMNS:
        out(f"  imagevote {name:7s} " + " ".join(f"{c[:6]} {val[('imagevote', sw, lv, c)]:5.1f}" for c in CLASSES))
    out("  change at 10 cm: " + " ".join(f"{c[:6]} {val[('imagevote', 'trans', 0.1, c)] - val[('imagevote', 'trans', 0.0, c)]:+.1f}" for c in CLASSES))
    heads = [n for _, names in deliv for n in names]
    out(""); out(f"== benchmark: the {len(heads)} heads trained with the delivered poses at the reference levels")
    for sw in ("trans", "rot", "drift"):
        lv = meta["reference"][sw]
        w = max((val[(n, sw, 0.0, "mean_all")] - val[(n, sw, lv, "mean_all")], n) for n in heads)
        e = max((abs(val[(n, sw, 0.0, "mean_electrical")] - val[(n, sw, lv, "mean_electrical")]), n) for n in heads)
        c = max((val[(n, sw, 0.0, k)] - val[(n, sw, lv, k)], n, k) for n in heads for k in CLASSES[2:])
        out(f"  {sw} {lv:g}: largest loss of mIoU {w[0]:.2f} ({w[1]}), of the electrical mean {e[0]:.2f} ({e[1]}), of one class {c[0]:.2f} ({c[1]}, {c[2]})")
    out(""); out("== benchmark: change of the head types against their clean value (mean over seeds)")
    for kind, lab, names in [("deliv.", lab, names) for lab, names in deliv] + [("aug.", lab, names) for lab, names in aug[:2]]:
        if names:
            for sw, lv, name in (("trans", 0.5, "50 cm"), ("rot", 1.0, "1 deg"), ("drift", 0.5, "drift 50")):
                out(f"  {kind:6s} {plain(lab):16s} {name:8s} " + " ".join(
                    f"{c[:6]} {g(names, sw, lv, c) - g(names, sw, 0.0, c):+5.1f}" for c in ["mean_all", "mean_electrical"] + thin + cover))
    if "lidar" in owner:
        for lab, names in deliv:
            if names:
                out(f"  deliv. {plain(lab):16s} mIoU above LiDAR-only: clean {g(names, 'trans', 0.0, 'mean_all') - val[('lidar', 'trans', 0.0, 'mean_all')]:+.2f}  "
                    f"50 cm {g(names, 'trans', 0.5, 'mean_all') - val[('lidar', 'trans', 0.0, 'mean_all')]:+.2f}  1 deg {g(names, 'rot', 1.0, 'mean_all') - val[('lidar', 'trans', 0.0, 'mean_all')]:+.2f}")
    cpath = os.path.join(res, f"protocol_{run}_confusion.json")
    concat = dict(deliv).get("Feature concat.", [])
    if os.path.exists(cpath) and concat:
        conf = json.load(open(cpath)); names = conf["class_names"]; ix = {n: i for i, n in enumerate(names)}
        out(""); out(f"== benchmark: confusion of the concatenation head, seeds pooled {concat}")
        for sw, lv, name in (("trans", 0.0, "clean"), ("trans", 0.5, "50 cm"), ("rot", 1.0, "1 deg")):
            cms = [r["cm"] for r in conf["rows"] if r["model"] in concat and r["sweep"] == sw and float(r["level"]) == lv]
            cm = [[sum(m[i][j] for m in cms) for j in range(len(names))] for i in range(len(names))]
            row = cm[ix["conductor"]]; tot = sum(row)
            prec = lambda c: cm[ix[c]][ix[c]] / sum(cm[i][ix[c]] for i in range(len(names)))
            out(f"  {name:6s} conductor points predicted as high_veg {100 * row[ix['high_veg']] / tot:.1f} %, low_veg {100 * row[ix['low_veg']] / tot:.1f} %, "
                f"herb {100 * row[ix['herb']] / tot:.1f} %, any other class {100 * (tot - row[ix['conductor']]) / tot:.1f} %; "
                f"precision conductor {100 * prec('conductor'):.2f} %, structural {100 * prec('structural'):.2f} %")
    banks = {}
    for name in sorted(os.listdir(res)):
        if name.startswith("bank_") and name.endswith(".json"):
            for sec, d in (json.load(open(os.path.join(res, name))).get("diagnostics") or {}).items():
                for head, e in d.items():
                    banks.setdefault(sec, {}).setdefault(head, e)
    if banks.get("val"):
        v, nl, ni, tr = (banks.get(k, {}) for k in ("val", "val_no_lidar", "val_no_image", "train"))
        out(""); out("== heads labs: clean accuracy (bank), water, modality ablations")
        if "lidar" in v and "imagevote" in v:
            for kind, (lab, names) in (("deliv.", deliv[0]), ("aug.", aug[0])):
                names = [n for n in names if n in v]
                if names:
                    m = lambda k: sum(v[n]["per_class"][k] for n in names) / len(names)
                    better = [k for k, _ in mt.NAMES if m(k) >= max(v["lidar"]["per_class"][k], v["imagevote"]["per_class"][k]) - 0.05]
                    worse = {k: round(m(k) - max(v["lidar"]["per_class"][k], v["imagevote"]["per_class"][k]), 1) for k, _ in mt.NAMES if k not in better}
                    out(f"  {kind} late: at or above the better input in {len(better)} of 11 classes; below it: {worse}")
                    out(f"    mIoU {sum(v[n]['mIoU'] for n in names) / len(names):.2f} = LiDAR-only {v['lidar']['mIoU']:.2f} {sum(v[n]['mIoU'] for n in names) / len(names) - v['lidar']['mIoU']:+.2f};"
                        f" OA {sum(v[n]['OA'] for n in names) / len(names):.2f}; per zone above LiDAR-only: "
                        + " ".join(f"{z} {sum(v[n]['per_zone'][z] for n in names) / len(names) - v['lidar']['per_zone'][z]:+.1f}" for z in v["lidar"]["per_zone"]))
        for kind, lab, names in [("deliv.", lab, names) for lab, names in deliv] + [("aug.", lab, names) for lab, names in aug]:
            names = [n for n in names if n in v]
            if names:
                out(f"  {kind:6s} {plain(lab):16s} water val " + " / ".join(f"{v[n]['per_class']['water']:.1f}" for n in names)
                    + ("   training zones " + " / ".join(f"{tr[n]['per_class']['water']:.1f}" for n in names if n in tr) if tr else "")
                    + ("   | without LiDAR: " + " ; ".join("cond %.1f struct %.1f insul %.1f" % tuple(nl[n]["per_class"][k] for k in thin[1:]) for n in names if n in nl) if nl else ""))
        if ni and "lidar" in v:
            worst = max((v["lidar"]["per_class"][k] - ni[n]["per_class"][k], n, k) for _, names in deliv + aug for n in names if n in ni for k in thin)
            out(f"  without image evidence: largest shortfall of an electrical class against LiDAR-only {worst[0]:.2f} ({worst[1]}, {worst[2]})")
        fragile = [n for lab, names in deliv if lab != "Late fusion" for n in names if n in nl]
        if len(fragile) >= 3:
            x = [sum(nl[n]["per_class"][k] for k in thin[1:]) / 3 for n in fragile]
            y = [val[(n, "rot", 0.0, "mean_electrical")] - val[(n, "rot", 1.0, "mean_electrical")] for n in fragile]
            out(f"  feature-level heads trained with the delivered poses ({len(fragile)}): correlation between the mean IoU of conductor, structural, "
                f"insulator WITHOUT LiDAR and the loss of the electrical mean at 1 deg: r = {_pearson(x, y):+.2f}")
            for n, a, b in sorted(zip(fragile, x, y), key=lambda t: t[1]):
                out(f"    {n:12s} without LiDAR {a:5.1f}   loss at 1 deg {b:5.1f}")
            st = [(nl[n]["per_class"]["structural"], n) for lab, names in aug if lab != "Late fusion" for n in names if n in nl]
            if st:
                out(f"  augmented feature-level heads ({len(st)}): structural cable without LiDAR " + " ".join(f"{n} {s_:.1f}" for s_, n in sorted(st)))
    fast = [(r, json.load(open(os.path.join(res, f"protocol_{r}_summary.json")))) for r in mt.FIX_RUNS
            if os.path.exists(os.path.join(res, f"protocol_{r}_summary.json"))]
    if fast and meta["mode"] == "exact":
        fval, fown = mt.merged_curves(fast)
        both = [m for m in owner if m in fown]
        d = [(fval[k] - val[k], k) for k in val if k[0] in both and k in fval and k[3] in ("mean_all", "mean_electrical")]
        hd = [x for x in d if x[1][0] not in ("lidar", "imagevote")]
        out(""); out(f"== fast form (runs {[r for r, _ in fast]}) minus exact form, same cameras: {len(both)} models in both")
        for cls in ("mean_all", "mean_electrical"):
            w = max(hd, key=lambda x: abs(x[0]) if x[1][3] == cls else -1)
            out(f"  fusion heads, {cls}: largest |difference| {abs(w[0]):.2f} ({w[1][0]}, {w[1][1]} {w[1][2]:g})")
            for sw in ("trans", "rot", "drift"):
                w = max((x for x in d if x[1][0] == "imagevote" and x[1][3] == cls and x[1][1] == sw), key=lambda x: abs(x[0]))
                out(f"  imagevote, {cls}, {sw}: largest difference {w[0]:+.2f} (level {w[1][2]:g})")


LT_COLUMNS = ("all", "pylon", "conductor", "structural", "insulator")
LT_LEVELS = [("none", 0.0, "delivered poses"), ("translation_cm", 10.0, "sigma_t 10 cm"), ("rotation_deg", 0.1, "sigma_r 0.1 deg"),
             ("translation_cm", 50.0, "sigma_t 50 cm"), ("rotation_deg", 1.0, "sigma_r 1 deg")]
VAL_ZONES = ("t1z5b", "t1z8", "t3z4", "t4z1", "t5b1", "t5b5")


def _lt_rows(path):
    """{(perturbation, level): {column: percent}} of a label-transfer table (gridnet.label_transfer.write_csv)."""
    import csv
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        out[(r["perturbation"], float(r["level"]))] = {k: (100 * float(v) if k not in ("perturbation", "level", "noise", "n", "images", "draws") else v)
                                                        for k, v in r.items()}
    return out


def motivating_report(res, out=print, run="label_transfer_v1"):
    """What Sec. 6.1 and the introduction quote: label transfer over every image with a mask of the validation zones."""
    folder = os.path.join(res, run)
    if not os.path.exists(os.path.join(folder, "label_transfer_val.csv")):
        return
    pooled = _lt_rows(os.path.join(folder, "label_transfer_val.csv"))
    zones = {z: _lt_rows(os.path.join(folder, f"label_transfer_{z}.csv")) for z in VAL_ZONES}
    clean = pooled[("none", 0.0)]
    out(""); out(f"== motivating experiment ({run}): share of the points that receive their own label [%], "
                 f"{clean['images']} images, error model {clean['noise']}, {clean['draws']} draw per image and level")
    out(f"  images per zone: " + ", ".join(f"{z} {zones[z][('none', 0.0)]['images']}" for z in VAL_ZONES)
        + f"; point-pixel pairs per image {float(clean['n']):,.0f}")
    for kind, lv, name in LT_LEVELS:
        out(f"  {name:16s} " + "  ".join(f"{c} {pooled[(kind, lv)][c]:.1f}" for c in LT_COLUMNS))
    for kind, lv, name in LT_LEVELS[1:3]:
        out(f"  {name}, change: " + "  ".join(f"{c} {pooled[(kind, lv)][c] - clean[c]:+.1f}" for c in LT_COLUMNS))
    for c in ("structural", "conductor"):
        r = pooled[("translation_cm", 10.0)]
        per = sorted((zones[z][("translation_cm", 10.0)][c], z) for z in VAL_ZONES)
        out(f"  {c} at sigma_t 10 cm: quartiles over the images {r[c + '_q25']:.1f} .. {r[c + '_q75']:.1f}; "
            f"zones {per[0][0]:.1f} ({per[0][1]}) .. {per[-1][0]:.1f} ({per[-1][1]})")
    thin = [pooled[(k, lv)][c] for k, lv in (("translation_cm", 50.0), ("rotation_deg", 1.0)) for c in ("conductor", "structural", "insulator")]
    out(f"  thin classes at 50 cm and 1 deg: {min(thin):.1f} .. {max(thin):.1f}")


def qualitative_report(res, out=print, folder="qualitative"):
    """What the qualitative figure and its paragraph quote: the pylon crops of the run (all of them from crops.json,
    the confusions of the two drawn ones from their files)."""
    import statistics
    table = os.path.join(res, folder, "crops.json")
    if not os.path.exists(table):
        return
    doc = json.load(open(table))
    share = {(r["file"], r["model"], r["sweep"], r["level"]): 100 * r["wrong"] for r in doc["rows"]}
    files = [c["file"] for c in doc["crops"]]
    cols = [("trans", 0.0, "delivered"), ("trans", 0.1, "10 cm"), ("trans", 0.5, "50 cm"), ("rot", 1.0, "1 deg")]
    out(""); out(f"== qualitative figure: {len(files)} pylons in {len({c['zone'] for c in doc['crops']})} zones, "
                 f"share of the power-line points with a wrong class [%]: min .. max (median)")
    models = list(dict.fromkeys(r["model"] for r in doc["rows"]))
    for m in models:
        cells = []
        for s_, lv, name in cols:
            v = [share[(f, m, s_, lv)] for f in files]
            cells.append(f"{name} {min(v):.1f} .. {max(v):.1f} ({statistics.median(v):.1f})")
        out(f"  {m:12s} " + "   ".join(cells))
    heads = [m for m in models if m not in ("lidar", "imagevote")]
    out(f"  fusion heads at 10 cm: largest |change| against the delivered poses "
        f"{max(abs(share[(f, m, 'trans', 0.1)] - share[(f, m, 'trans', 0.0)]) for f in files for m in heads):.2f}")
    for m in ("late_a", "concat_a", "win5_a", "late_nat_b1"):
        if m in models:
            out(f"  {m}: largest increase at 1 deg over the delivered poses {max(share[(f, m, 'rot', 1.0)] - share[(f, m, 'trans', 0.0)] for f in files):+.2f}")
    order = sorted((share[(f, "concat_n", "rot", 1.0)], f) for f in files)
    out("  concat_n at 1 deg, sorted: " + ", ".join(f"{f[:-4]} {v:.1f}" for v, f in order))
    try:
        from gridnet import qualitative as Q
    except ImportError:                                  # numpy missing: the table above is all there is
        return
    names = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb", "rock_soil", "road", "water", "building"]
    for f in files:
        path = os.path.join(res, folder, f)
        if not os.path.exists(path):
            continue                                     # only the crops of the figure are kept in the repository
        crop = Q.load_crop(path)
        rank = [x for _, x in order].index(f) + 1
        out(f"  figure row {f[:-4]} (rank {rank} of {len(files)} at 1 deg): lidar {share[(f, 'lidar', 'trans', 0.0)]:.1f}, concat_n delivered "
            f"{share[(f, 'concat_n', 'trans', 0.0)]:.1f} / 1 deg {share[(f, 'concat_n', 'rot', 1.0)]:.1f}, concat_a 1 deg {share[(f, 'concat_a', 'rot', 1.0)]:.1f}")
        cm = crop["cm"][Q.key("concat_n", "rot", 1.0)]
        rec = [100 * cm[i, i] / cm[i].sum() for i in range(4)]
        parts = []
        for i in (1, 2):
            row = cm[i].copy(); row[i] = 0
            j = int(row.argmax())
            parts.append(f"{names[i]} -> {names[j]} {100 * row[j] / cm[i].sum():.1f} %")
        out(f"    concat_n at 1 deg: right class for pylon {rec[0]:.1f}, conductor {rec[1]:.1f}, structural {rec[2]:.1f}, insulator {rec[3]:.1f}; " + ", ".join(parts))


def image_branch_report(res, out=print, name="upernet_v2_eval2d.json"):
    """Sec. IV-B: validation of the 2D network on its fixed crops (a copy of eval2d.json of the run on Drive, without
    the confusion matrix, plus the lines of its config.json that say how the crops are drawn)."""
    path = os.path.join(res, name)
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    pc, c = d["per_class"], d.get("config", {})
    out(""); out(f"== image branch ({d['run']}, {d['ckpt']} at iteration {d['it']}): {d['val_crops']} fixed crops = "
                 f"{len(d['val_zones'])} zones ({', '.join(d['val_zones'])}) x {c.get('val_images')} images x {c.get('val_crops_per_image')} crops")
    out(f"  mIoU {d['mIoU']:.1f}  OA {d['OA']:.1f}   (mean of the {len(pc)} per-class values: {sum(pc.values()) / len(pc):.2f})")
    out("  " + "  ".join(f"{k} {v:.1f}" for k, v in pc.items()))


def leaderboard_report(res, out=print, tag="leaderboard_v1"):
    """Sec. VI-F / VI-G: the check of the leaderboard run on a validation zone (per voxel against per raw point) and what
    predicting the test zones cost."""
    cp, rp = os.path.join(res, f"{tag}_check_t1z8.json"), os.path.join(res, f"{tag}_run.json")
    if not os.path.exists(cp):
        return
    d = json.load(open(cp))
    out(""); out(f"== leaderboard run, check on validation zone {d['zone']} ({d['fused']['points']:,} raw points, {d['fused']['voxels']:,} voxels)")
    out(f"  2D export of {d['export2d']['images']} images against the stored cache: same class on {100 * d['export2d']['same_class']:.3f} % of the pixels")
    for m, name in (("fused", "fused model"), ("lidar", "LiDAR-only")):
        r = d[m]
        diff = {k: r["raw_per_class"][k] - r["voxel_per_class"][k] for k in r["raw_per_class"]}
        order = sorted(diff, key=lambda k: -abs(diff[k]))
        out(f"  {name:11s} validation run {d['validation_run'][m]:.2f}, reproduced {d['predicted'][m]:.2f};  per voxel {r['voxel_mIoU']:.2f} mIoU / "
            f"{r['voxel_OA']:.2f} OA, per raw point {r['raw_mIoU']:.2f} / {r['raw_OA']:.2f}  (raw - voxel {r['raw_mIoU'] - r['voxel_mIoU']:+.2f} mIoU)")
        out("              largest per-class differences, raw - voxel: " + ", ".join(f"{k} {diff[k]:+.2f}" for k in order[:3]))
    out(f"  fused prediction identical to the validation run on {100 * d['same_voxels']:.3f} % of the voxels")
    if os.path.exists(rp):
        run = json.load(open(rp))
        t, z = run["totals"], run["zones"]
        out(f"  nine test zones: {t['images']:,} images, {t['points']:,} raw points, {t['voxels']:,} voxels in {t['minutes']:.1f} min "
            f"(2D outputs {t['export2d_minutes']:.1f}, prediction {t['predict_minutes']:.1f}) + check {run['check_minutes']:.1f} min = "
            f"{(t['minutes'] + run['check_minutes']) / 60:.2f} h; {60 * t['export2d_minutes'] / t['images']:.2f} s per image; per zone "
            f"{min(v['minutes'] for v in z.values()):.1f} .. {max(v['minutes'] for v in z.values()):.1f} min")
        out(f"  points without a view: {min(v['no_view_percent'] for v in z.values()):.1f} .. {max(v['no_view_percent'] for v in z.values()):.1f} %; "
            f"water predicted by the LiDAR-only model: at most {max(v['predicted_percent']['lidar']['water'] for v in z.values()):.2f} % of a zone, "
            f"by the fused model: {max(v['predicted_percent']['fused']['water'] for v in z.values()):.2f} %")


def test_set_report(res, out=print):
    """Sec. VI-F, abstract, introduction, conclusion: our two leaderboard entries against the published baselines, the
    other entries of the leaderboard and our validation numbers."""
    path = os.path.join(res, "test_set.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    ours = {r["ours"]: r for r in d["rows"] if r.get("ours")}
    if any(r["miou"] is None for r in ours.values()):
        out(""); out("== test set: our two rows are not entered yet"); return
    f, l = ours["fused"], ours["lidar"]
    pub = [r for r in d["rows"] if r.get("cite")]
    best = max(pub, key=lambda r: r["miou"])
    snap = d["leaderboard_snapshot"]
    others = [e for e in snap if e[0] not in (f["entry"], l["entry"])]
    out(""); out(f"== hidden test zones, leaderboard read on {d['leaderboard_accessed']} ({len(snap)} entries)")
    out(f"  LiDAR-only {l['miou']:.2f} mIoU / {l['oa']:.2f} OA   fused {f['miou']:.2f} / {f['oa']:.2f}   gain of fusion {f['miou'] - l['miou']:+.2f} mIoU, {f['oa'] - l['oa']:+.2f} OA")
    out(f"  fused against the best published baseline ({best['method']}, {best['miou']:.2f}): {f['miou'] - best['miou']:+.2f}; against each published row: "
        + ", ".join(f"{r['method']} {f['miou'] - r['miou']:+.2f}" for r in pub))
    out("  LiDAR-only against the published LiDAR-only rows: " + ", ".join(
        f"{r['method']} {l['miou'] - r['miou']:+.2f} mIoU / {l['oa'] - r['oa']:+.2f} OA" for r in pub if r["method"].startswith(("SPT", "PTv3"))))
    top = max(others, key=lambda e: e[2])
    out(f"  best other entry: {top[0]} {top[2]:.2f}; rank of the fused model by mIoU {1 + sum(e[2] > f['miou'] for e in others)}, "
        f"by accuracy {1 + sum(e[3] > f['oa'] for e in others)} (best accuracy {max(e[3] for e in others):.2f})")
    ev = os.path.join(res, "fusion_late_a_eval.json")
    if os.path.exists(ev):
        v = json.load(open(ev))
        out(f"  validation, full zones per voxel: fused {v['mIoU']:.2f}, LiDAR-only {v['lidar_only']['mIoU']:.2f} (gain {v['mIoU'] - v['lidar_only']['mIoU']:+.2f}); "
            f"test minus validation: fused {f['miou'] - v['mIoU']:+.2f}, LiDAR-only {l['miou'] - v['lidar_only']['mIoU']:+.2f}")


def shared_report(res, out=print, shared_run=None, iid_run=None):
    """Shared against independent pose errors (Sec. VI-E, Table "shared"): what the text quotes beyond the table."""
    from gridnet.stats import paired_summary
    shared_run, iid_run = shared_run or mt.SHARED_RUN, iid_run or mt.EXACT_RUN
    ps_, pi_ = (os.path.join(res, f"protocol_{r}_summary.json") for r in (shared_run, iid_run))
    if not (os.path.exists(ps_) and os.path.exists(pi_)):
        return
    sh, ii = json.load(open(ps_)), json.load(open(pi_))
    a = {(c["model"], c["sweep"], round(float(c["level"]), 6), c["class"]): c for c in sh["curves"]}
    b = {(c["model"], c["sweep"], round(float(c["level"]), 6), c["class"]): c for c in ii["curves"]}
    A = lambda m, sw, lv, cls: 100 * a[(m, sw, lv, cls)]["iou_mean"]
    B = lambda m, sw, lv, cls: 100 * b[(m, sw, lv, cls)]["iou_mean"]
    names = sh["meta"]["class_names"]
    cols = mt.shared_columns(sh, ii)                                  # (i.i.d. sweep, shared sweep, level): reference and largest
    ref = [c for c in cols if abs(c[2] - ii["meta"].get("reference", mt.REFERENCE)[c[0]]) < 1e-9]
    top = [c for c in cols if c not in ref]
    tag = lambda base, lv: (f"{100 * lv:g} cm" if base == "trans" else f"{lv:g} deg")
    types = [(("aug. " if "augmentation" in blk else "deliv. ") + lab.replace("$", "").replace("{\\times}", "x"), ms)
             for blk, lab, ms in mt.shared_values(sh, ii)[1] if blk]
    deliv = [m for t, ms in types if t.startswith("deliv.") for m in ms]
    aug = [m for t, ms in types if t.startswith("aug.") for m in ms]
    pairs = list(zip(deliv, aug))                                     # same head type and seed: the order of FIX_GROUPS
    mean = lambda xs: sum(xs) / len(xs)
    out(""); out(f"== shared against independent pose errors: runs {shared_run} / {iid_run} ({sh['meta']['mode']}, {sh['meta']['draws']} draws; "
                 f"{len(deliv)} heads trained with the delivered poses, {len(aug)} with pose noise)")
    worst = max(abs(A(m, sw, 0.0, g) - B(m, base, 0.0, g)) for m in deliv + aug + ["lidar", "imagevote"] for base, sw, _ in cols[:1] for g in ("mean_all", "mean_electrical"))
    out(f"  delivered poses: largest difference between the two runs {worst:.3f}")
    for g, title in (("mean_all", "mIoU"), ("mean_electrical", "electrical")):
        out(f"  {title}: clean | " + " | ".join(f"{tag(base, lv)} ind / shared" for base, sw, lv in cols))
        for t, ms in [("LiDAR-only", ["lidar"]), ("ImageVote", ["imagevote"])] + types:
            out(f"    {t:24s} {mean([A(m, cols[0][1], 0.0, g) for m in ms]):6.2f} | "
                + " | ".join(f"{mean([B(m, base, lv, g) for m in ms]):6.2f} / {mean([A(m, sw, lv, g) for m in ms]):6.2f}" for base, sw, lv in cols))
    # every head trained with the delivered poses loses more when the error is shared
    for group, heads in (("deliv.", deliv), ("aug.", aug)):
        for g, title in (("mean_electrical", "electrical"), ("mean_all", "mIoU")):
            for base, sw, lv in cols:
                d = [A(m, sw, lv, g) - B(m, base, lv, g) for m in heads]
                out(f"  {group} shared minus independent, {title}, {tag(base, lv)}: mean {mean(d):+.2f}  range {min(d):+.2f} .. {max(d):+.2f}  lower in {sum(x < 0 for x in d)} of {len(d)}")
    # losses at the shared reference levels
    for group, heads in (("deliv.", deliv), ("aug.", aug)):
        la = max((A(m, sw, 0.0, "mean_all") - A(m, sw, lv, "mean_all"), m, sw, lv) for m in heads for _, sw, lv in ref)
        le = max((A(m, sw, 0.0, "mean_electrical") - A(m, sw, lv, "mean_electrical"), m, sw, lv) for m in heads for _, sw, lv in ref)
        lc = max((A(m, sw, 0.0, c) - A(m, sw, lv, c), m, c, sw, lv) for m in heads for _, sw, lv in ref for c in names)
        out(f"  {group} shared reference levels: largest loss of mIoU {la[0]:.2f} ({la[1]}, {la[2]} {la[3]:g}), of the electrical mean {le[0]:.2f} "
            f"({le[1]}, {le[2]} {le[3]:g}), of one class {lc[0]:.2f} ({lc[1]}, {lc[2]}, {lc[3]} {lc[4]:g})")
        for _, sw, lv in ref:
            x = max((A(m, sw, 0.0, "mean_all") - A(m, sw, lv, "mean_all"), m) for m in heads)
            out(f"    {sw} {lv:g}: largest loss of mIoU {x[0]:.2f} ({x[1]})")
    for t, ms in types:
        for base, sw, lv in ref[:1]:
            cls = ("mean_all", "mean_electrical", "conductor", "structural", "insulator")
            out(f"  {t:24s} loss at shared {tag(base, lv)}: " + "  ".join(f"{c} {mean([A(m, sw, 0.0, c) - A(m, sw, lv, c) for m in ms]):.2f}" for c in cls)
                + "   independent: " + "  ".join(f"{c} {mean([B(m, base, 0.0, c) - B(m, base, lv, c) for m in ms]):.2f}" for c in cls[:2]))
    # the largest levels
    for t, ms in types:
        out(f"  {t:24s} loss of the electrical mean, shared (independent): " + "  ".join(
            f"{tag(base, lv)} {mean([A(m, sw, 0.0, 'mean_electrical') - A(m, sw, lv, 'mean_electrical') for m in ms]):.2f} "
            f"({mean([B(m, base, 0.0, 'mean_electrical') - B(m, base, lv, 'mean_electrical') for m in ms]):.2f})" for base, sw, lv in top)
            + "   mIoU minus LiDAR-only, shared: " + "  ".join(f"{tag(base, lv)} {mean([A(m, sw, lv, 'mean_all') for m in ms]) - A('lidar', sw, lv, 'mean_all'):+.2f}" for base, sw, lv in top))
    x = max((A(m, sw, 0.0, "mean_electrical") - A(m, sw, lv, "mean_electrical"), m, sw, lv) for m in aug for sw in sh["meta"]["sweeps"] for lv in [round(float(v), 6) for v in sh["meta"]["sweeps"][sw]])
    out(f"  augmented heads ({len(aug)}): largest loss of the electrical mean over all levels of the shared sweeps: {x[0]:.2f} ({x[1]}, {x[2]} {x[3]:g})")
    vals = [A(m, sw, lv, "mean_electrical") for m in aug for _, sw, lv in top]
    out(f"  augmented heads, electrical mean at the largest shared levels: {min(vals):.2f} .. {max(vals):.2f} over heads; head types "
        + " / ".join(f"{mean([A(m, sw, lv, 'mean_electrical') for m in ms]):.2f}" for _, sw, lv in top for t, ms in types if t.startswith("aug.")))
    for t, ms in types:
        if t.startswith("aug."):
            out(f"  {t:24s} shared (independent) at the largest levels: " + "  ".join(
                f"{c} " + " / ".join(f"{mean([A(m, sw, lv, c) for m in ms]):.1f} ({mean([B(m, base, lv, c) for m in ms]):.1f})" for base, sw, lv in top)
                for c in ("mean_all", "road", "water", "building")) + f"   clean road {mean([A(m, top[0][1], 0.0, 'road') for m in ms]):.1f}")
    d = [mean([B(m, base, lv, "mean_all") for m in ms]) - mean([A(m, sw, lv, "mean_all") for m in ms]) for t, ms in types if t.startswith("aug.") for base, sw, lv in top]
    out(f"  augmented head types, mIoU at the largest levels: independent minus shared {min(d):.2f} .. {max(d):.2f}")
    # ImageVote
    for base, sw, lv in ref:
        out(f"  imagevote at {tag(base, lv)}: loss of the electrical mean shared {A('imagevote', sw, 0.0, 'mean_electrical') - A('imagevote', sw, lv, 'mean_electrical'):.2f} / "
            f"independent {B('imagevote', base, 0.0, 'mean_electrical') - B('imagevote', base, lv, 'mean_electrical'):.2f}; "
            f"of mIoU {A('imagevote', sw, 0.0, 'mean_all') - A('imagevote', sw, lv, 'mean_all'):.2f} / {B('imagevote', base, 0.0, 'mean_all') - B('imagevote', base, lv, 'mean_all'):.2f}")
    # pairs: augmentation against delivered poses under shared errors; window sizes
    for base, sw, lv in top:
        s_ = paired_summary([A(y, sw, lv, "mean_electrical") - A(x_, sw, lv, "mean_electrical") for x_, y in pairs], "greater")
        out(f"  seed pairs, augmentation minus delivered, electrical, shared {tag(base, lv)}: n={s_['n']} mean {s_['mean']:+.2f} range {s_['min']:+.2f} .. {s_['max']:+.2f} "
            f"positive {s_['positive']}/{s_['n']} p {s_['p']:.4f}")
    by_type = dict(types)
    for other in ("Window 3x3", "Window 5x5", "Late fusion"):
        if "deliv. " + other in by_type and "deliv. Feature concat." in by_type:
            for base, sw, lv in top:
                d = [A(y, sw, lv, "mean_electrical") - A(x_, sw, lv, "mean_electrical") for x_, y in zip(by_type["deliv. Feature concat."], by_type["deliv. " + other])]
                di = [B(y, base, lv, "mean_electrical") - B(x_, base, lv, "mean_electrical") for x_, y in zip(by_type["deliv. Feature concat."], by_type["deliv. " + other])]
                out(f"  deliv. {other} minus concatenation, electrical, {tag(base, lv)}: shared mean {mean(d):+.2f} range {min(d):+.2f} .. {max(d):+.2f} positive {sum(v > 0 for v in d)}/{len(d)}"
                    f"   independent mean {mean(di):+.2f} positive {sum(v > 0 for v in di)}/{len(di)}")
    # zones as pairs
    zs, zi = (os.path.join(res, f"protocol_{r}_zones.json") for r in (shared_run, iid_run))
    if os.path.exists(zs) and os.path.exists(zi):
        zv, ze = mt.zone_values_dict(json.load(open(zs))), mt.zone_values_dict(json.load(open(zi)))
        zones = sorted({k[3] for k in zv})
        zmean = lambda D, ms, sw, lv, z, c: mean([D[(m, sw, lv, z, c)] for m in ms])
        for base, sw, lv in top:
            d = [zmean(zv, aug, sw, lv, z, "mean_electrical") - zmean(zv, deliv, sw, lv, z, "mean_electrical") for z in zones]
            out(f"  zones, augmentation minus delivered ({len(pairs)} head pairs per zone), electrical, shared {tag(base, lv)}: mean {mean(d):+.2f} range {min(d):+.2f} .. {max(d):+.2f} positive {sum(v > 0 for v in d)}/{len(d)}")
        for base, sw, lv in ref[:1] + top:
            d = [zmean(zv, deliv, sw, lv, z, "mean_electrical") - zmean(ze, deliv, base, lv, z, "mean_electrical") for z in zones]
            out(f"  zones, deliv. heads, shared minus independent, electrical, {tag(base, lv)}: mean {mean(d):+.2f} range {min(d):+.2f} .. {max(d):+.2f} lower in {sum(v < 0 for v in d)} of {len(d)}")
    # spread over the draws: one shared error decides for a whole zone
    for group, heads in (("deliv.", deliv), ("aug.", aug)):
        x = max((100 * a[(m, sw, lv, "mean_electrical")]["iou_std"], m, sw, lv) for m in heads for _, sw, lv in cols)
        y = max((100 * b[(m, base, lv, "mean_electrical")]["iou_std"], m, base, lv) for m in heads for base, _, lv in cols)
        out(f"  {group} std over the draws of the electrical mean: shared up to {x[0]:.2f} ({x[1]}, {x[2]} {x[3]:g}), independent up to {y[0]:.2f} ({y[1]}, {y[2]} {y[3]:g})")
    for t, ms in types:
        if t.startswith("deliv."):
            out(f"  {t:24s} std over the draws, shared, mean over seeds: " + "  ".join(
                f"{tag(base, lv)} {mean([100 * a[(m, sw, lv, 'mean_electrical')]['iou_std'] for m in ms]):.1f}" for base, sw, lv in cols))


def main():
    res = os.path.join(ROOT, "results")
    runs = sys.argv[1:]
    summaries = ([(r, json.load(open(os.path.join(res, f"protocol_{r}_summary.json")))) for r in runs] if runs else mt.fix_summaries(res))
    if not summaries:
        sys.exit("no protocol summary in results/")
    report(summaries)
    if not runs:
        image_branch_report(res)
        benchmark_report(summaries, res)
        motivating_report(res)
        qualitative_report(res)
        shared_report(res)
        leaderboard_report(res)
        test_set_report(res)


if __name__ == "__main__":
    main()
