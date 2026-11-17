"""Robustness protocol: evaluate LiDAR-only, ImageVote and trained fusion heads under camera-pose error.

  python scripts/run_protocol.py --cache /content/cache --cache2d /content/cache2d --data /content/data \
      --assoc-dir $DRIVE/fusion/assoc_upernet_convnext_t_v2 --litept /content/LitePT \
      --backbone-ckpt /content/backbone_best.pth --zones t1z5b,t1z8,t3z4,t4z1,t5b1,t5b5 \
      --heads late_v1=$DRIVE/fusion/late_v1/best.pth concat_v1=$DRIVE/fusion/concat_v1/best.pth \
      --out $DRIVE/protocol/fast_v1 --subset-dir $DRIVE/protocol/subsets --mode fast --draws 10

Per zone the frozen-backbone features of a class-stratified subset are computed once (`--subset-dir`, shared by
every later protocol run), then every (sweep, level, draw) costs one re-projection + one head forward per model
(measured on an A100, run fast_v1: ~1 s per zone-draw for 6 models, 18 min for 6 zones x 133 evaluations).
`--mode exact` rebuilds the association (re-rendered depth maps, re-selected views) per draw instead (~6 s per
zone-draw): the mode of the headline numbers. Draws are seeded by sweep name and level value, so two runs share
cameras for the draws they share and `scripts/compare_protocol.py` compares them pair by pair. The i.i.d. sweeps
take `--draws` random draws per level, the drift one evaluation per direction. `--sweeps trans_shared,rot_shared`
are the shared-error sweeps: one translation / one rotation for all images of a zone per draw, with the levels and
draws of the i.i.d. sweeps (`scripts/compare_shared.py` puts such a run next to the i.i.d. run). Results: <out>/results.json (weighted
confusion per model/sweep/level/draw/zone; pixel displacement and coverage per zone/sweep/level/draw) with a gzip
copy, summary.json (curves, scores, displacement per level), curves.csv, scores.csv, confusion.json (one pooled
confusion matrix per model/sweep/level). A results.json is continued (zones added) only with the same models, sweeps,
draws and mode - otherwise the script stops and asks for another --out.

`--dump-pred DIR` also saves, for draw 0 of the levels in `--dump-levels`, the predictions around the largest pylons of
every zone (gridnet/qualitative.py: one small file per pylon) - the input of scripts/plot_qualitative.py. Draw 0 of a
level has the same cameras in every run with the same --seed, so the crops show predictions that are part of the
benchmark numbers. A zone that is already in <out>/results.json is skipped and therefore not dumped again.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gridnet import qualitative  # noqa: E402
from gridnet.train.engine_fusion import FrozenBackbone, load_zone_context  # noqa: E402
from gridnet.train.protocol import SWEEPS, ProtocolModels, run_protocol  # noqa: E402


def parse_levels(text, default):
    return [float(x) for x in text.split(",")] if text else list(default)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True); ap.add_argument("--cache2d", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--assoc-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--subset-dir", default=None, help="where <zone>/subset.npz live (default: --out)")
    ap.add_argument("--litept", default=None); ap.add_argument("--backbone-ckpt", default=None)
    ap.add_argument("--zones", default="t1z5b,t1z8,t3z4,t4z1,t5b1,t5b5")
    ap.add_argument("--heads", nargs="*", default=[], help="name=path/to/best.pth (head type and K come from the checkpoint)")
    ap.add_argument("--no-imagevote", action="store_true"); ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--sweeps", default="trans,rot,drift", help="of " + ", ".join(SWEEPS))
    ap.add_argument("--levels-trans", default=None); ap.add_argument("--levels-rot", default=None); ap.add_argument("--levels-drift", default=None)
    ap.add_argument("--levels-trans-shared", default=None); ap.add_argument("--levels-rot-shared", default=None)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--mode", choices=["fast", "fast_retest", "exact"], default="fast",
                    help="fast = stored views re-projected, visibility kept; fast_retest = also tested against the clean "
                         "depth maps; exact = association rebuilt under the perturbed cameras")
    ap.add_argument("--cap", type=int, default=500_000, help="subset: max points per class")
    ap.add_argument("--cap-full", type=int, default=2_000_000, help="subset: max points per electrical class")
    ap.add_argument("--radius", type=float, default=25.0); ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--eval-max-points", type=int, default=900_000)
    ap.add_argument("--views", type=int, default=8, help="views per point at test time: the nearest of the stored ones "
                    "(clean level, fast modes) or of the re-association (exact); the heads were trained with 8")
    ap.add_argument("--stored-views", type=int, default=8, help="views per point of the association archive and of the subsets")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--assoc-chunk", type=int, default=20_000_000)
    ap.add_argument("--assoc-source", choices=["auto", "archive", "compute"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dump-pred", default=None, help="folder for crops of the predictions around the largest pylons of every zone (figures)")
    ap.add_argument("--dump-levels", default="trans:0,trans:0.5,rot:1", help="sweep:level pairs to keep (draw 0 of each); levels of this run")
    ap.add_argument("--dump-models", default="", help="models to keep, comma-separated (default: every model of the run)")
    ap.add_argument("--dump-radius", type=float, default=35.0, help="horizontal radius of a crop around its pylon [m]")
    ap.add_argument("--dump-pylons", type=int, default=2, help="crops per zone (largest pylons first)")
    ap.add_argument("--dump-voxel", type=float, default=0.05, help="one point per voxel of this size is kept for drawing [m]")
    ap.add_argument("--dump-max-points", type=int, default=300_000, help="most points kept per crop")
    a = ap.parse_args()

    zones = [z for z in a.zones.split(",") if z]
    sweeps = {}
    for s in a.sweeps.split(","):
        s = s.strip()
        if s:
            if s not in SWEEPS:
                raise SystemExit(f"[protocol] STOPPED: unknown sweep {s!r} (known: {', '.join(SWEEPS)})")
            sweeps[s] = parse_levels(getattr(a, f"levels_{s}", None), SWEEPS[s])
    specs = dict(h.split("=", 1) for h in a.heads)
    heads = ProtocolModels.from_checkpoints(specs, device=a.device)
    models = ProtocolModels(heads, device=a.device, imagevote=not a.no_imagevote, lidar=not a.no_lidar)
    backbone = None
    if a.backbone_ckpt:
        backbone = FrozenBackbone(a.litept, a.backbone_ckpt, device=a.device)
    cfg = {"radius": a.radius, "grid": a.grid, "eval_max_points": a.eval_max_points}

    def zone_loader(zone):
        # the clean association always has the stored number of views: --views only selects among them (run_protocol)
        ctx = load_zone_context(a.cache, a.cache2d, a.data, a.assoc_dir, zone, k=a.stored_views, tau=a.tau, chunk=a.assoc_chunk,
                                need_depths=a.mode == "fast_retest", with_feat=True, device=a.device,
                                assoc_source=a.assoc_source)
        with np.load(os.path.join(a.cache, zone, "points.npz")) as p:
            normals = p["normals"] if "normals" in p else None
        return ctx, normals

    dump = None
    if a.dump_pred:
        levels = qualitative.parse_levels(a.dump_levels)
        bad = [f"{s}:{lv:g}" for s, lv in levels if s not in sweeps or not any(abs(lv - x) < 1e-9 for x in sweeps[s])]
        if bad:
            raise SystemExit(f"[protocol] STOPPED: --dump-levels {bad} are not levels of this run ({sweeps})")
        wanted = [m.strip() for m in a.dump_models.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in models.names]
        if unknown:
            raise SystemExit(f"[protocol] STOPPED: --dump-models {unknown} are not models of this run ({list(models.names)})")
        dump = qualitative.CropDump(a.dump_pred, levels, models=wanted, radius=a.dump_radius, pylons=a.dump_pylons,
                                    voxel=a.dump_voxel, max_points=a.dump_max_points, seed=a.seed)
        spath = os.path.join(a.dump_pred, "dump.json")                   # crops of one folder share their settings
        new = json.loads(json.dumps(dump.settings()))
        if os.path.exists(spath):
            old = json.load(open(spath))
            if old != new:
                raise SystemExit(f"[protocol] STOPPED: {a.dump_pred} holds crops made with other settings ({old}); use another folder")
        else:
            with open(spath, "w") as fh:
                json.dump(new, fh, indent=1)
    os.makedirs(a.out, exist_ok=True)
    if not os.path.exists(os.path.join(a.out, "results.json")):          # the arguments of the call that started the run
        with open(os.path.join(a.out, "args.json"), "w") as fh:
            json.dump(vars(a), fh, indent=1)
    try:
        run_protocol(models, zone_loader, zones, a.out, sweeps, draws=a.draws, mode=a.mode, tau=a.tau, seed=a.seed, k=a.views,
                     subset_dir=a.subset_dir, backbone=backbone, cfg=cfg, cap=a.cap, cap_full=a.cap_full, on_draw=dump)
    except ValueError as e:
        if "other settings" not in str(e):
            raise
        raise SystemExit(f"[protocol] STOPPED: {e}")
    print(f"[protocol] done -> {a.out}/summary.json, curves.csv, scores.csv", flush=True)
    if dump is not None:
        files = sorted(f for f in os.listdir(a.dump_pred) if f.endswith(".npz") and "_crop" in f)
        size = sum(os.path.getsize(os.path.join(a.dump_pred, f)) for f in files) / 2 ** 20
        print(f"[protocol] crops -> {a.dump_pred}: {len(files)} files, {size:.1f} MB ({len(dump.zones)} zones in this call)", flush=True)


if __name__ == "__main__":
    main()
