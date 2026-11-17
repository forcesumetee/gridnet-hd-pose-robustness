"""Train many fusion-head variants at once on precomputed frozen-backbone features (see gridnet/train/bank.py).

  python scripts/train_heads_bank.py --cache /content/cache --cache2d /content/cache2d --data /content/data \
      --assoc-dir $DRIVE/fusion/assoc_upernet_convnext_t_v2 --subset-dir /content/banks \
      --subset-archive $DRIVE/protocol/subsets_upernet_convnext_t_v2 \
      --litept /content/LitePT --backbone-ckpt /content/backbone_best.pth --out-root $DRIVE/fusion \
      --specs "late_b:late concat_b:concat concat_ld50:concat:lidar_drop=0.5 concat_lp:concat:lidar_input=prob" \
      --references late_v1,concat_v1 --diagnostics --tag lab1

Steps: (1) the feature bank of every training / validation zone is taken from --subset-dir, else copied from
--subset-archive, else built with the frozen backbone (one inference pass over the zone, ~3-5 min on an A100) and
archived; (2) all heads of --specs are trained side by side, validated on the val banks after every pass, and
their best weights written to <out-root>/<name>/best.pth with eval_bank/eval.json; (3) with --diagnostics the best
heads are also evaluated without LiDAR evidence, without image evidence and on the training-zone banks. A summary of
the whole group goes to <out-root>/bank_<tag>.json.

A head spec is name:head[:key=value,...] with head = late | concat | window and keys k, lidar_input (feat | prob),
image_input (feat | prob, concat only), lidar_drop, natural_frac, pose_t, pose_r, aug_prob, seed, lr, wd,
uniform_views. Heads with the same seed (and sampling / augmentation settings) see the same batches. Runs are
resumable per pass: finished heads are skipped; a name is never reused for another configuration.
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gridnet.train.bank import (BankTrainer, bank_class_counts, check_names, format_counts, format_table,  # noqa: E402
                                load_bank_context, parse_specs, result_metrics, zone_class_counts)
from gridnet.train.engine_fusion import FrozenBackbone, class_weights_from_counts, load_zone_context  # noqa: E402
from gridnet.train.protocol import ProtocolModels, ensure_subset  # noqa: E402


def ensure_banks(a, zones):
    """Local bank of every zone: present -> keep; in the archive -> copy; else build with the backbone and archive."""
    backbone = None
    cfg = {"radius": a.radius, "grid": a.grid, "eval_max_points": a.eval_max_points}
    for z in zones:
        local = os.path.join(a.subset_dir, z, "subset.npz")
        arch = os.path.join(a.subset_archive, z, "subset.npz") if a.subset_archive else None
        if os.path.exists(local):
            continue
        t0 = time.time()
        if arch and os.path.exists(arch):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            shutil.copyfile(arch, local + ".tmp"); os.replace(local + ".tmp", local)
            print(f"[bank] {z}: copied from the archive ({os.path.getsize(local) / 1e9:.2f} GB, {(time.time() - t0) / 60:.1f} min)", flush=True)
            continue
        if backbone is None:
            assert a.backbone_ckpt, f"no bank for {z} and no --backbone-ckpt to build it"
            backbone = FrozenBackbone(a.litept, a.backbone_ckpt, device=a.device)
        ctx = load_zone_context(a.cache, a.cache2d, a.data, a.assoc_dir, z, k=a.views, tau=a.tau, chunk=a.assoc_chunk,
                                need_depths=False, with_feat=False, device=a.device, verbose=False, with_maps=False)
        ensure_subset(ctx, local, backbone, cfg, seed=a.subset_seed, cap=a.cap, cap_full=a.cap_full)
        ctx.free()
        if arch and not os.path.exists(arch):
            os.makedirs(os.path.dirname(arch), exist_ok=True)
            shutil.copyfile(local, arch + ".tmp"); os.replace(arch + ".tmp", arch)
        print(f"[bank] {z}: built ({os.path.getsize(local) / 1e9:.2f} GB, {(time.time() - t0) / 60:.1f} min)"
              + (" and archived" if arch else ""), flush=True)
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="LiDAR caches (needed to build banks; pose-noise augmentation reads the point coordinates)")
    ap.add_argument("--cache2d", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--assoc-dir", required=True)
    ap.add_argument("--subset-dir", required=True, help="local folder of the banks: <zone>/subset.npz")
    ap.add_argument("--subset-archive", default=None, help="Drive folder with the same layout (read when present, written once)")
    ap.add_argument("--litept", default=None); ap.add_argument("--backbone-ckpt", default=None)
    ap.add_argument("--out-root", required=True, help="<out-root>/<name>/ per head (GridNet/fusion)")
    ap.add_argument("--specs", default="", help="head specs: name:head[:key=value,...] separated by spaces")
    ap.add_argument("--references", default="", help="comma-separated names of existing heads (<out-root>/<name>/best.pth, "
                    "e.g. the crop-trained late_v1) that are evaluated on the banks next to the new ones")
    ap.add_argument("--tag", default="lab", help="summary file <out-root>/bank_<tag>.json")
    ap.add_argument("--train-zones", default="t1z6a,t2z5,t3z3,t5a1,t5b2,t5c1,t6z2,t6z4")
    ap.add_argument("--val-zones", default="t1z5b,t1z8,t3z4,t4z1,t5b1,t5b5")
    ap.add_argument("--passes", type=int, default=2); ap.add_argument("--its-per-zone", type=int, default=300)
    ap.add_argument("--batch-points", type=int, default=120_000); ap.add_argument("--chunk", type=int, default=60_000)
    ap.add_argument("--warmup", type=float, default=0.05); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lovasz", type=float, default=1.0); ap.add_argument("--class-weight-power", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=500_000); ap.add_argument("--cap-full", type=int, default=2_000_000)
    ap.add_argument("--subset-seed", type=int, default=0, help="must match the protocol's --seed so both use one subset")
    ap.add_argument("--radius", type=float, default=25.0); ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--eval-max-points", type=int, default=900_000)
    ap.add_argument("--views", type=int, default=8); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--assoc-chunk", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=0, help="zone order (head seeds are set per spec)")
    ap.add_argument("--diagnostics", action="store_true", help="modality ablations on val + evaluation on the training banks")
    ap.add_argument("--build-only", action="store_true", help="only make sure the banks exist")
    ap.add_argument("--max-hours", type=float, default=1e9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    train_zones = [z for z in a.train_zones.split(",") if z]
    val_zones = [z for z in a.val_zones.split(",") if z]
    cfg = {"train_zones": train_zones, "val_zones": val_zones, "passes": a.passes, "its_per_zone": a.its_per_zone,
           "batch_points": a.batch_points, "chunk": a.chunk, "warmup": a.warmup,
           "clip": a.clip, "lovasz": a.lovasz, "class_weight_power": a.class_weight_power, "tau": a.tau, "views": a.views,
           "radius": a.radius, "grid": a.grid, "eval_max_points": a.eval_max_points, "max_points": 600_000,
           "head_points": a.batch_points, "head_chunk": a.chunk, "seed_order": a.seed}
    specs = parse_specs(a.specs)
    ref_names = [x.strip() for x in a.references.split(",") if x.strip()]
    if not a.build_only:                                # everything that can be wrong with the request, before the slow steps
        assert specs, "no head specs given (--specs)"
        clash = sorted(set(ref_names) & {s.name for s in specs})
        assert not clash, f"{clash}: a head cannot be trained and be a reference at the same time"
        check_names(specs, cfg, a.out_root)
    os.makedirs(a.subset_dir, exist_ok=True)
    t0 = time.time()
    ensure_banks(a, train_zones + val_zones)
    print(f"[bank] banks ready for {len(train_zones)} training + {len(val_zones)} validation zones ({(time.time() - t0) / 60:.1f} min)", flush=True)
    counts = zone_class_counts(a.subset_dir, train_zones + val_zones)
    print("[bank] thousands of points per class (training zones, then validation zones):\n" + format_counts(counts), flush=True)
    if a.build_only:
        return
    class_w = class_weights_from_counts(bank_class_counts(a.subset_dir, train_zones), a.class_weight_power)
    print("[bank] class weights:", [round(float(w), 2) for w in class_w], flush=True)
    trainer = BankTrainer(specs, class_w, cfg, a.out_root, device=a.device)
    for r in trainer.runs:
        print(f"[bank] head {r.spec.name}: {r.spec.head}" + (f" K={r.spec.k}" if r.spec.head == "window" else "")
              + (f" lidar_input={r.spec.lidar_input}" if r.spec.head != "late" else "")
              + (f" image_input={r.spec.image_input}" if r.spec.image_input != "feat" else "")
              + (f" lidar_drop={r.spec.lidar_drop}" if r.spec.lidar_drop else "")
              + (f" natural_frac={r.spec.natural_frac}" if r.spec.natural_frac != 1.0 else "")
              + (f" pose noise {r.spec.pose_t} m / {r.spec.pose_r} deg" if (r.spec.pose_t or r.spec.pose_r) else "")
              + f" seed {r.spec.seed}: {sum(p.numel() for p in r.head.parameters()) / 1e3:.1f} k parameters", flush=True)
    refs = {}
    for n in ref_names:
        path = os.path.join(a.out_root, n, "best.pth")
        if os.path.exists(path):
            refs[n] = path
        else:
            print(f"[bank] reference head {n}: no {path} - skipped", flush=True)
    if refs:
        trainer.add_references(ProtocolModels.from_checkpoints(refs, device=a.device))
        print(f"[bank] reference heads (evaluated only): {', '.join(refs)}", flush=True)

    def loader(zone, need_xyz, with_feat):
        return load_bank_context(a.cache, a.cache2d, a.data, a.subset_dir, zone, need_xyz=need_xyz,
                                 with_feat=with_feat, device=a.device, chunk=a.assoc_chunk, verbose=False)

    t1 = time.time()
    status = trainer.fit(loader, max_hours=a.max_hours)
    print(f"[bank] training {status} after {(time.time() - t1) / 60:.1f} min", flush=True)
    if status != "done":
        return
    order = ["lidar", "imagevote"] + list(refs) + [s.name for s in specs]
    summary = {"tag": a.tag, "specs": [vars(s) for s in specs], "references": list(refs), "cfg": cfg,
               "best": {r.spec.name: r.best for r in trainer.runs},
               "class_counts": {z: [float(v) for v in c] for z, c in counts.items()}}
    if a.diagnostics:
        diag = trainer.diagnostics(loader, train_zones)
        summary["diagnostics"] = diag
        titles = {"val": "validation zones, best weights (bank estimate)", "val_no_lidar": "validation, heads WITHOUT LiDAR evidence",
                  "val_no_image": "validation, heads WITHOUT image evidence", "train": "training zones (zones both backbones were trained on)"}
        for case in ("val", "val_no_lidar", "val_no_image", "train"):
            if case in diag:
                print(f"\n=== {titles[case]} ===\n" + format_table(diag[case], order), flush=True)
    else:
        trainer.load_best()
        res = trainer.evaluate(loader, val_zones, verbose=False)
        val = {n: result_metrics(d) for n, d in res.items()}
        summary["diagnostics"] = {"val": val}
        print("\n=== validation zones, best weights (bank estimate) ===\n" + format_table(val, order), flush=True)
    with open(os.path.join(a.out_root, f"bank_{a.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\n[bank] done in {(time.time() - t0) / 60:.1f} min -> {a.out_root}/<name>/best.pth, eval_bank/, bank_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
