"""Shared against independent pose errors: a protocol run with the sweeps trans_shared / rot_shared next to the run
with the i.i.d. sweeps trans / rot of the same models (pure Python: reads two summary.json).

  python scripts/compare_shared.py --shared $DRIVE/protocol/shared_v1 --iid $DRIVE/protocol/exact_v1 \
      [--out compare.json]

In a shared-error sweep every image of a zone gets the SAME translation or rotation (one draw per zone and draw,
from the distribution of the i.i.d. sweep), in the i.i.d. sweep every image its own. Levels and number of draws are
the same, so a row of the table answers one question: does it matter whether the cameras share the error?

Printed: (1) a check - the delivered poses (level 0) must give both runs the same numbers, they evaluate the same
heads on the same points; (2) per error model and level the mean IoU over the electrical classes and over all
classes, independent | shared, mean over the training seeds of a head type, with the standard deviation over the
draws of the shared run (one shared error per zone is a far noisier draw than one error per image); (3) per class,
shared minus independent at the largest level; (4) how far a projected point moves in the image in both runs.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402  (head types and their seeds: FIX_GROUPS)

SHARED = {"trans_shared": "trans", "rot_shared": "rot"}       # shared sweep -> its i.i.d. counterpart
UNIT = {"trans": (100.0, "cm"), "rot": (1.0, "deg")}
GROUPS = ("mean_electrical", "mean_all")
CLASSES = [k for k, _ in mt.NAMES]


def load_summary(path):
    """summary.json of a run folder, or the file itself."""
    if os.path.isdir(path):
        path = os.path.join(path, "summary.json")
    with open(path) as fh:
        return json.load(fh)


def plain(label):
    """Row label of the tables without LaTeX."""
    return re.sub(r"\$(\d)\{\\times\}(\d)\$", r"\1x\2", label)


def model_groups(shared, iid):
    """[(block, label, [models])]: LiDAR-only and ImageVote, then the head types of the fix table with the seeds that
    BOTH runs hold (a head type without any common seed is left out)."""
    have = set(shared["meta"]["models"]) & set(iid["meta"]["models"])
    out = [("", "LiDAR-only", ["lidar"]), ("", "ImageVote", ["imagevote"])]
    block = ""
    for label, names in mt.FIX_GROUPS:
        if names is None:
            block = label
            continue
        out.append((block, plain(label), list(names)))
    return [(b, l, [n for n in names if n in have]) for b, l, names in out if any(n in have for n in names)]


def _index(summary):
    return {(c["model"], c["sweep"], round(float(c["level"]), 6), c["class"]): c for c in summary["curves"]}


def _mean(values):
    v = [x for x in values if x is not None and x == x]
    return sum(v) / len(v) if v else float("nan")


def clean_check(shared, iid):
    """Largest difference between the two runs with the delivered poses, in points of mean IoU, over the models they
    share (both groups of classes); None when they share no model."""
    a, b = _index(shared), _index(iid)
    worst = None
    for m in sorted(set(shared["meta"]["models"]) & set(iid["meta"]["models"])):
        for g in GROUPS:
            va = [a[k]["iou_mean"] for k in a if k[0] == m and k[2] == 0.0 and k[3] == g]
            vb = [b[k]["iou_mean"] for k in b if k[0] == m and k[2] == 0.0 and k[3] == g]
            if va and vb:
                d = max(abs(100 * x - 100 * vb[0]) for x in va)
                worst = d if worst is None else max(worst, d)
    return worst


def shared_rows(shared, iid):
    """One row per (head type, shared sweep, level, class or class group): the mean over the seeds both runs hold of
    the IoU [%] with independent and with shared errors, and of its standard deviation over the draws."""
    a, b = _index(shared), _index(iid)
    rows = []
    for block, label, names in model_groups(shared, iid):
        for sw, base in SHARED.items():
            if sw not in shared["meta"]["sweeps"] or base not in iid["meta"]["sweeps"]:
                continue
            common = sorted(set(round(float(x), 6) for x in shared["meta"]["sweeps"][sw]) & set(round(float(x), 6) for x in iid["meta"]["sweeps"][base]))
            for lv in common:
                for cls in list(GROUPS) + CLASSES:
                    ks = [(n, sw, lv, cls) for n in names]; ki = [(n, base, lv, cls) for n in names]
                    if not all(k in a for k in ks) or not all(k in b for k in ki):
                        continue
                    rows.append({"block": block, "head": label, "n": len(names), "models": names, "sweep": sw, "iid_sweep": base,
                                 "level": lv, "class": cls,
                                 "iid": 100 * _mean([b[k]["iou_mean"] for k in ki]), "shared": 100 * _mean([a[k]["iou_mean"] for k in ks]),
                                 "iid_std_draws": 100 * _mean([b[k]["iou_std"] for k in ki]),
                                 "shared_std_draws": 100 * _mean([a[k]["iou_std"] for k in ks]),
                                 "draws_shared": min(a[k]["draws"] for k in ks), "draws_iid": min(b[k]["draws"] for k in ki)})
    return rows


def report(shared, iid, out=print, names=("shared", "independent")):
    rows = shared_rows(shared, iid)
    ms, mi = shared["meta"], iid["meta"]
    out(f"== {names[0]}: {ms['mode']} mode, {ms['draws']} draws, {len(ms['models'])} models, zones {ms.get('zones')}")
    out(f"== {names[1]}: {mi['mode']} mode, {mi['draws']} draws, {len(mi['models'])} models, zones {mi.get('zones')}")
    if ms["mode"] != mi["mode"] or ms.get("views", 8) != mi.get("views", 8) or sorted(ms.get("zones", [])) != sorted(mi.get("zones", [])):
        out("   NOTE: the two runs differ in mode, views or zones - the comparison is not like with like")
    worst = clean_check(shared, iid)
    if worst is None:
        out("check: the runs share no model")
    else:
        out(f"check, delivered poses: largest difference between the runs over the models they share: {worst:.3f} points of mean IoU "
            + ("(same heads, same points: as it must be)" if worst < 0.05 else "-> NOT the same clean numbers: other heads, points or code?"))
    heads = list(dict.fromkeys((r["block"], r["head"], r["n"]) for r in rows))
    for sw, base in SHARED.items():
        scale, unit = UNIT[base]
        levels = sorted({r["level"] for r in rows if r["sweep"] == sw and r["level"] > 0})
        if not levels:
            continue
        for g, title in (("mean_electrical", "mean IoU over the electrical classes"), ("mean_all", "mean IoU over all classes")):
            out("")
            out(f"-- {'translation' if base == 'trans' else 'rotation'} error, {title} [%]: independent | shared (+- std over the draws of the shared run)")
            out(f"{'':30s}{'n':>2s}  " + "  ".join(f"{f'{scale * lv:g} {unit}':^20s}" for lv in levels))
            block = None
            for b, h, n in heads:
                if b != block and b:
                    out(f"  {b}")
                block = b
                cells = []
                for lv in levels:
                    r = [x for x in rows if x["head"] == h and x["block"] == b and x["sweep"] == sw and x["level"] == lv and x["class"] == g]
                    cells.append(f"{r[0]['iid']:5.1f} | {r[0]['shared']:5.1f} +-{r[0]['shared_std_draws']:4.1f}" if r else f"{'--':^20s}")
                out(f"  {('  ' if b else '') + h:28s}{n:>2d}  " + "  ".join(cells))
        top = levels[-1]
        out("")
        out(f"-- {'translation' if base == 'trans' else 'rotation'} error of {scale * top:g} {unit}, per class: shared minus independent [IoU points]")
        out(f"{'':30s}    " + " ".join(f"{c[:8]:>8s}" for c in CLASSES))
        block = None
        for b, h, n in heads:
            if b != block and b:
                out(f"  {b}")
            block = b
            vals = []
            for c in CLASSES:
                r = [x for x in rows if x["head"] == h and x["block"] == b and x["sweep"] == sw and x["level"] == top and x["class"] == c]
                d = r[0]["shared"] - r[0]["iid"] if r else float("nan")
                vals.append(f"{d:+8.1f}" if d == d else f"{'--':>8s}")          # a class no zone of the run contains
            out(f"  {('  ' if b else '') + h:28s}    " + " ".join(vals))
    da = {(d["sweep"], round(float(d["level"]), 6)): d for d in shared.get("displacement", [])}
    db = {(d["sweep"], round(float(d["level"]), 6)): d for d in iid.get("displacement", [])}
    lines = []
    for sw, base in SHARED.items():
        scale, unit = UNIT[base]
        for lv in sorted({k[1] for k in da if k[0] == sw and k[1] > 0}):
            x, y = da.get((sw, lv)), db.get((base, lv))
            if x and y and x.get("shift_px") is not None and y.get("shift_px") is not None:
                lines.append(f"  {scale * lv:g} {unit}: independent {y['shift_px']:.1f} px, shared {x['shift_px']:.1f} px"
                             f"   (electrical classes {y['shift_px_electrical']:.1f} / {x['shift_px_electrical']:.1f})")
    if lines:
        out("")
        out("-- median displacement of a projected point in the image (full-resolution pixels)")
        for l in lines:
            out(l)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shared", required=True, help="run with trans_shared / rot_shared: its folder or summary.json")
    ap.add_argument("--iid", required=True, help="run with the i.i.d. sweeps trans / rot of the same models")
    ap.add_argument("--names", default="shared,independent")
    ap.add_argument("--out", default=None, help="write the rows (and the clean check) as JSON")
    a = ap.parse_args()
    shared, iid = load_summary(a.shared), load_summary(a.iid)
    rows = report(shared, iid, names=tuple(a.names.split(",")))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"shared": a.shared, "iid": a.iid, "clean_check": clean_check(shared, iid), "rows": rows}, fh, indent=1)
        print(f"\nwrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
