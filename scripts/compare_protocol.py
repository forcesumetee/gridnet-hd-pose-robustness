"""Compare two protocol runs on the evaluations they share (same model, sweep, level, draw, zone = same cameras),
typically the fast approximation against the exact re-association on a few zones and draws.

  python scripts/compare_protocol.py --a $DRIVE/protocol/fast_v1 --b $DRIVE/protocol/fast_v1_exact --out $DRIVE/protocol/fast_v1_exact/compare.json

Prints, per model / sweep / level, the mean IoU over all classes and over the electrical classes for both runs (%),
their difference (b - a), the class that differs most and the coverage of both; writes the rows as JSON.
"""
import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gridnet.train.protocol import compare_runs  # noqa: E402


def load_results(run_dir):
    """results.json of a run directory, or its gzip copy when only that one is there."""
    path = os.path.join(run_dir, "results.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    with gzip.open(path + ".gz", "rt") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="first run directory (results.json), e.g. the fast run")
    ap.add_argument("--b", required=True, help="second run directory, e.g. the exact run")
    ap.add_argument("--names", default="fast,exact", help="labels of the two runs in the table")
    ap.add_argument("--out", default=None, help="JSON file for the rows (default: <b>/compare.json)")
    a = ap.parse_args()
    ra, rb = load_results(a.a), load_results(a.b)
    rows = compare_runs(ra, rb)
    assert rows, "the two runs share no (model, sweep, level, draw, zone) evaluation"
    na, nb = (a.names.split(",") + ["a", "b"])[:2]
    out = a.out or os.path.join(a.b, "compare.json")
    with open(out, "w") as fh:
        json.dump({"a": a.a, "b": a.b, "names": [na, nb], "rows": rows}, fh, indent=1)
    zones = sorted({z for r in rows for z in r["zones"]}); nd = max(len(r["draws"]) for r in rows)
    print(f"{len(rows)} (model, sweep, level) cells on zones {zones}, up to {nd} shared draws per level -> {out}")
    print(f"{'model':14s} {'sweep':6s} {'level':>6s}  {'all ' + na:>9s} {'all ' + nb:>9s} {'diff':>6s}   {'elec ' + na:>10s} {'elec ' + nb:>10s} {'diff':>6s}   "
          f"{'largest class difference':28s} {'cover ' + na:>10s} {'cover ' + nb:>10s}")
    for r in rows:
        print(f"{r['model']:14s} {r['sweep']:6s} {r['level']:6g}  {100 * r['mean_all_a']:9.2f} {100 * r['mean_all_b']:9.2f} "
              f"{100 * (r['mean_all_b'] - r['mean_all_a']):+6.2f}   {100 * r['mean_electrical_a']:10.2f} {100 * r['mean_electrical_b']:10.2f} "
              f"{100 * (r['mean_electrical_b'] - r['mean_electrical_a']):+6.2f}   {r['max_diff_class']:12s} {100 * r['max_diff']:+7.2f}         "
              f"{100 * r['coverage_a']:10.1f} {100 * r['coverage_b']:10.1f}")


if __name__ == "__main__":
    main()
