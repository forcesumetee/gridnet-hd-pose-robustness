"""Label transfer under camera-pose error over all images of a zone (the experiment behind the motivating figure).

  python scripts/label_transfer_zone.py --cache <cache root> --data <dataset root> --zone t5b5 --out <out root>
  python scripts/label_transfer_zone.py --out <out root> --combine t1z5b,t1z8,t3z4,t4z1,t5b1,t5b5

For every image of the zone the points it sees receive the label of the mask pixel they project to, once with the
delivered pose and then with perturbed poses (gridnet/label_transfer.py has the definitions). Needs no GPU and no
network: the 5 cm points of the cache (<cache root>/<zone>/points.npz: xyz, origin, labels), the masks and the poses
(<dataset root>/<zone>/masks/*.png, <dataset root>/<zone>/pose/).

Outputs in <out root>/<zone>/:
  records.jsonl          one line per finished image (resume: finished images are skipped)
  label_transfer.csv     pooled share per error kind and level, all classes (+ quartiles over the images)
  label_transfer.json    settings, number of images / points, elapsed time, the rows of the CSV
--combine pools the records of several zones into <out root>/<name>/ (default name: all).

The default error model is the one of the robustness protocol (--noise protocol: sigma per axis); --noise fixed is
the first single-image experiment (a displacement of fixed length, a rotation by a fixed angle). One draw per image
and level is the default: with n images a level is the pooled result of n independent draws.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_ZONE = {}          # arrays of the zone, set before the worker processes are forked


def _settings(a):
    return {"zone": a.zone, "noise": a.noise, "seed": a.seed, "draws": a.draws, "trans_cm": a.trans, "rot_deg": a.rot,
            "max_dist": a.max_dist, "ds": a.ds, "tau": a.tau}


def _one_image(image_id):
    from gridnet import label_transfer as lt
    from gridnet.io import load_mask
    z = _ZONE
    mask = load_mask(z["zone_dir"], image_id)
    s = z["settings"]
    return image_id, lt.image_records(z["xyz"], z["labels"], mask, z["poses"][image_id], z["cal"], image_id,
                                      trans_cm=s["trans_cm"], rot_deg=s["rot_deg"], draws=s["draws"], seed=s["seed"],
                                      noise=s["noise"], max_dist=s["max_dist"], ds=s["ds"], tau=s["tau"])


def read_records(path):
    """{image id: records} of a records.jsonl; a truncated last line (interrupted run) is dropped."""
    done = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                done[d["image"]] = d["records"]
    return done


def write_summary(out_dir, records, settings, extra):
    from gridnet import label_transfer as lt
    rows = lt.summarize(records)
    lt.write_csv(os.path.join(out_dir, "label_transfer.csv"), rows, settings["noise"])
    doc = {"settings": settings, **extra, "rows": rows}
    with open(os.path.join(out_dir, "label_transfer.json.tmp"), "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(os.path.join(out_dir, "label_transfer.json.tmp"), os.path.join(out_dir, "label_transfer.json"))
    return rows


def show(rows, title, out=print):
    out(title)
    out(f"  {'perturbation':<15}{'level':>7}{'all':>8}{'pylon':>8}{'conduct.':>9}{'struct.':>9}{'insul.':>8}{'images':>8}")
    for r in rows:
        out(f"  {r['perturbation']:<15}{r['level']:>7g}" + "".join(f"{100 * r[k]:>{w}.1f}" for k, w in (
            ("all", 8), ("pylon", 8), ("conductor", 9), ("structural", 9), ("insulator", 8))) + f"{r['images']:>8d}")


def run_zone(a):
    if a.workers != 1:                      # numpy's own threads would fight the worker processes
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(k, "1")
    import numpy as np
    sys.path.insert(0, ROOT)
    from gridnet.camera import Pose, load_zone_cameras

    out_dir = os.path.join(a.out, a.zone)
    os.makedirs(out_dir, exist_ok=True)
    settings = _settings(a)
    meta_path, rec_path = os.path.join(out_dir, "settings.json"), os.path.join(out_dir, "records.jsonl")
    if a.force:
        for p in (meta_path, rec_path):
            if os.path.exists(p):
                os.remove(p)
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if old != settings:
            diff = {k: (old.get(k), settings[k]) for k in settings if old.get(k) != settings[k]}
            sys.exit(f"{out_dir} holds records of other settings {diff}: use another --out or --force")
    else:
        json.dump(settings, open(meta_path, "w"), indent=1)

    zone_dir = os.path.join(a.data, a.zone)
    cal, poses_utm = load_zone_cameras(zone_dir)
    with np.load(os.path.join(a.cache, a.zone, "points.npz")) as p:
        xyz, origin, labels = p["xyz"], p["origin"].astype(np.float64), p["labels"]
    poses = {k: Pose(S=v.S - origin, R=v.R, opk=v.opk) for k, v in poses_utm.items()}
    have = sorted(k for k in poses if os.path.exists(os.path.join(zone_dir, "masks", k + ".png")))
    if a.images:
        want = [x for x in a.images.split(",") if x]
        missing = [x for x in want if x not in have]
        if missing:
            sys.exit(f"no mask or pose for {missing}")
        ids = want
    else:
        ids = have[::a.stride]
        ids = ids[:a.max_images] if a.max_images else ids
    if not ids:
        sys.exit(f"no image with a mask and a pose in {zone_dir}")
    done = read_records(rec_path)
    todo = [i for i in ids if i not in done]
    print(f"[{a.zone}] {len(xyz):,} points, {len(ids)} images ({len(have)} with a mask), {len(done)} done before, "
          f"{len(todo)} to do; noise {a.noise}, {a.draws} draw(s) per level", flush=True)

    _ZONE.update(xyz=xyz, labels=labels, poses=poses, cal=cal, zone_dir=zone_dir, settings=settings)
    # worker processes are forked; outside Linux (macOS: BLAS and fork do not mix safely) the default is one process
    workers = a.workers or (min(os.cpu_count() or 1, 8) if sys.platform.startswith("linux") else 1)
    t0 = time.time()
    with open(rec_path, "a") as fh:
        def keep(image_id, records, k):
            fh.write(json.dumps({"image": image_id, "records": records}) + "\n"); fh.flush()
            done[image_id] = records
            if k % a.log_every == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[{a.zone}] {k}/{len(todo)} images  {el / 60:.1f} min  ({el / k:.1f} s per image)", flush=True)

        if workers > 1 and len(todo) > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(min(workers, len(todo))) as pool:
                for k, (image_id, records) in enumerate(pool.imap_unordered(_one_image, todo), 1):
                    keep(image_id, records, k)
        else:
            for k, image_id in enumerate(todo, 1):
                keep(*_one_image(image_id), k)
    records = [r for i in ids for r in done[i]]
    rows = write_summary(out_dir, records, settings, {"images": len(ids), "images_with_mask": len(have), "points": int(len(xyz)),
                                                      "elapsed_min": (time.time() - t0) / 60, "workers": workers, "image_ids": ids})
    show(rows, f"[{a.zone}] share of the points that receive their own label [%], {len(ids)} images")
    print(f"[{a.zone}] wrote {out_dir}/label_transfer.csv", flush=True)


def combine(a):
    sys.path.insert(0, ROOT)
    zones = [z for z in a.combine.split(",") if z]
    records, settings, images = [], None, 0
    for z in zones:
        d = os.path.join(a.out, z)
        s = json.load(open(os.path.join(d, "settings.json")))
        s.pop("zone")
        if settings is not None and s != settings:
            sys.exit(f"{z} was run with other settings than {zones[0]}: {s} vs {settings}")
        settings = s
        done = read_records(os.path.join(d, "records.jsonl"))
        ids = json.load(open(os.path.join(d, "label_transfer.json")))["image_ids"]    # the images of the zone's last finished run
        missing = [i for i in ids if i not in done]
        if missing:
            sys.exit(f"{z}: no records for {len(missing)} of its {len(ids)} images")
        images += len(ids)
        for image_id in ids:
            for r in done[image_id]:
                records.append({**r, "image": f"{z}/{image_id}"})
    out_dir = os.path.join(a.out, a.name)
    os.makedirs(out_dir, exist_ok=True)
    rows = write_summary(out_dir, records, {**settings, "zone": "+".join(zones)}, {"images": images, "zones": zones})
    show(rows, f"[{a.name}] {len(zones)} zones, {images} images: share of the points that receive their own label [%]")
    print(f"[{a.name}] wrote {out_dir}/label_transfer.csv", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output root; one folder per zone")
    ap.add_argument("--zone", help="zone to run")
    ap.add_argument("--cache", help="cache root with <zone>/points.npz")
    ap.add_argument("--data", help="dataset root with <zone>/masks and <zone>/pose")
    ap.add_argument("--combine", help="comma list of finished zones to pool instead of running one")
    ap.add_argument("--name", default="all", help="folder of the pooled result (--combine)")
    ap.add_argument("--noise", default="protocol", choices=["protocol", "fixed"])
    ap.add_argument("--draws", type=int, default=1, help="draws per image and level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trans", type=lambda s: [float(x) for x in s.split(",") if x], default=[2.0, 5.0, 10.0, 25.0, 50.0], help="levels in cm")
    ap.add_argument("--rot", type=lambda s: [float(x) for x in s.split(",") if x], default=[0.05, 0.1, 0.25, 0.5, 1.0], help="levels in degrees")
    ap.add_argument("--stride", type=int, default=1, help="every n-th image")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--images", default="", help="comma list of image ids instead of --stride / --max-images")
    ap.add_argument("--workers", type=int, default=0, help="processes (0 = one per core, at most 8, on Linux; 1 elsewhere)")
    ap.add_argument("--max-dist", type=float, default=150.0)
    ap.add_argument("--ds", type=int, default=4, help="depth-map downscale factor of the visibility test")
    ap.add_argument("--tau", type=float, default=0.5, help="depth tolerance of the visibility test (m)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="discard the records of this zone and start again")
    a = ap.parse_args()
    if a.combine:
        return combine(a)
    if not (a.zone and a.cache and a.data):
        ap.error("--zone, --cache and --data are needed to run a zone")
    run_zone(a)


if __name__ == "__main__":
    main()
