"""Paired comparisons behind Table "stats" of the paper, and the same comparisons with validation ZONES as pairs.

  python scripts/paper_stats.py                                   # seeds as pairs, from the summaries behind the tables
  python scripts/paper_stats.py --results exact_v1/results.json.gz [more files]   # + zones as pairs
  python scripts/paper_stats.py --results exact_v1/results.json.gz --write-zones results/protocol_exact_v1_zones.json
                                                                  # + the per-zone values the statistics table reads

Seeds as pairs: two heads that differ in one thing and share the seed (scripts/make_tables.py: stat_comparisons).
Zones as pairs: for each validation zone the metric of a group of heads (mean over its seeds and over the draws of a
level) minus that of the other group -> six paired values. A zone's "electrical" mean runs over the electrical
classes that occur in the zone. The per-zone confusion matrices come from the results.json(.gz) of protocol runs
(GridNet/protocol/<tag>/); a model is taken from the first file that has it.
Test: exact Wilcoxon signed-rank (gridnet/stats.py).
"""
import argparse
import gzip
import importlib.util
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from gridnet.stats import paired_summary  # noqa: E402

_spec = importlib.util.spec_from_file_location("make_tables", os.path.join(ROOT, "scripts", "make_tables.py"))
mt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mt)
ELECTRICAL = ("pylon", "conductor", "structural", "insulator")
ZONE_METRICS = ["mean_all", "mean_electrical", "mean_wo_water"]


def load_results(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def iou_per_class(cm):
    """Per-class IoU of a confusion matrix given as nested lists (rows = ground truth); None where the union is empty."""
    n = len(cm)
    row = [sum(cm[i]) for i in range(n)]
    col = [sum(cm[i][j] for i in range(n)) for j in range(n)]
    return [(cm[i][i] / (row[i] + col[i] - cm[i][i]) if row[i] + col[i] - cm[i][i] > 0 else None) for i in range(n)], row


def zone_values(results_list, cls):
    """{(model, sweep, level, zone): metric in %} : per draw the zone's IoU, then the mean over the draws.
    cls: a class name, "mean_all" (classes present in the zone), "mean_electrical" or "mean_wo_water"."""
    acc, seen = {}, {}
    for res in results_list:
        names = res["meta"]["class_names"]
        for r in res["runs"]:
            if seen.setdefault(r["model"], id(res)) != id(res):
                continue                                             # a model belongs to the first file that has it
            iou, row = iou_per_class(r["cm"])
            present = [i for i in range(len(names)) if row[i] > 0 and iou[i] is not None]
            if cls == "mean_all":
                idx = present
            elif cls == "mean_electrical":
                idx = [i for i in present if names[i] in ELECTRICAL]
            elif cls == "mean_wo_water":
                idx = [i for i in present if names[i] != "water"]
            else:
                idx = [i for i in present if names[i] == cls]
            if idx:
                acc.setdefault((r["model"], r["sweep"], float(r["level"]), r["zone"]), []).append(100 * sum(iou[i] for i in idx) / len(idx))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def zone_pairs(results_list, first, second, sweep, level, cls):
    """Paired differences over zones: mean over the heads of `second` minus mean over the heads of `first`."""
    val = zone_values(results_list, cls)
    zones = sorted({k[3] for k in val})
    out = []
    for z in zones:
        a = [val[(m, sweep, level, z)] for m in first if (m, sweep, level, z) in val]
        b = [val[(m, sweep, level, z)] for m in second if (m, sweep, level, z) in val]
        if len(a) == len(first) and len(b) == len(second) and a:
            out.append((z, sum(b) / len(b) - sum(a) / len(a)))
    return out


def zone_doc(results_list, run=None):
    """Per-zone values of ZONE_METRICS for every (model, sweep, level) of the results files: the content of
    results/protocol_<run>_zones.json, from which scripts/make_tables.py makes the zone part of the statistics table."""
    vals = {m: zone_values(results_list, m) for m in ZONE_METRICS}
    keys = sorted(set().union(*[set(v) for v in vals.values()]))
    meta = results_list[0]["meta"]
    return {"note": "Per validation zone: mean IoU in % over the classes present in the zone (mean_all), over its electrical "
                    "classes (mean_electrical) and over all its classes but water (mean_wo_water); per draw the IoU of the "
                    "zone, then the mean over the draws. Written by scripts/paper_stats.py --write-zones from the "
                    "results.json(.gz) of the protocol run.",
            "run": run, "mode": meta.get("mode"), "draws": meta.get("draws"), "zones": sorted({k[3] for k in keys}),
            "metrics": ZONE_METRICS,
            "rows": [[k[0], k[1], k[2], k[3]] + [None if k not in vals[m] else round(vals[m][k], 4) for m in ZONE_METRICS] for k in keys]}


def line(label, s):
    if not s["n"]:
        return f"  {label:44s} no pair"
    return (f"  {label:44s} n={s['n']:2d}  mean {s['mean']:+6.2f}  range {s['min']:+6.2f} .. {s['max']:+6.2f}  "
            f"positive {s['positive']}/{s['n']}  p {s['p']:.4f} ({s['alternative']})")


def plain(label):
    return label.replace("$", "").replace("\\sigma_t", "sigma_t").replace("\\sigma_r", "sigma_r").replace("\\,", " ").replace("^\\circ", " deg").replace("{\\times}", "x")


def report(summaries, results_list=(), out=print):
    out(f"seeds as pairs - runs {[r for r, _ in summaries]}, mode {summaries[0][1]['meta']['mode']}")
    for title, alt, rows in mt.stat_rows(summaries):
        out(plain(title))
        for label, s in rows:
            out(line(plain(label), s))
    if not results_list:
        return
    out(""); out("zones as pairs - models of the given results files")
    doc = zone_doc(results_list)
    for title, alt, k, rows in mt.zone_stat_rows(mt.zone_values_dict(doc)):
        out(plain(title) + f"  [{k} head pair(s) averaged per zone]")
        for label, s in rows:
            out(line(plain(label), s))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="protocol runs with a summary in results/ (default: those behind the tables)")
    ap.add_argument("--results", nargs="*", default=[], help="results.json(.gz) files of protocol runs: adds the comparisons with zones as pairs")
    ap.add_argument("--write-zones", default=None, help="write the per-zone values of --results to this file (results/protocol_<run>_zones.json)")
    a = ap.parse_args()
    res = os.path.join(ROOT, "results")
    summaries = ([(r, json.load(open(os.path.join(res, f"protocol_{r}_summary.json")))) for r in a.runs] if a.runs else mt.fix_summaries(res))
    if not summaries:
        sys.exit("no protocol summary in results/")
    doc = report(summaries, [load_results(p) for p in a.results])
    if a.write_zones:
        if not doc:
            sys.exit("--write-zones needs --results")
        doc["run"] = summaries[0][0] if len(summaries) == 1 else None
        with open(a.write_zones, "w") as fh:
            json.dump(doc, fh, separators=(",", ":"))
        print(f"wrote {a.write_zones}: {len(doc['rows'])} rows, zones {doc['zones']}")


if __name__ == "__main__":
    main()
