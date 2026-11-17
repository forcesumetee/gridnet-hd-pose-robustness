"""One-time: turn the per-image 2D cache of some zones (fp16 or compact npz files on Drive) into one uncompressed
tar per zone of COMPACT npz members, stored next to the zone folders:

    <cache2d>/<zone>/<image>.npz      (as written by train_upernet.py --export-cache; untouched)
    <cache2d>/<zone>.tar              (new: members <zone>/<image>.npz, uint8 prob + int8 feat + feat_scale)
    <cache2d>/<zone>.tar.json         (new: {"zone", "n", "ids", "bytes"})

Why: Colab reads Drive through FUSE at ~10-15 MB/s when it is one small file after another, but at >100 MB/s for
one big sequential file. The fusion notebooks stream-extract <zone>.tar (~24 MB/image) to local disk instead of
copying hundreds of fp16 files (~50 MB each), which turns hours of copying per session into minutes.

Needs numpy only (no GPU, no torch): run it on a Colab CPU runtime with the Drive mounted, e.g.

    python scripts/compact_cache2d.py --cache2d /content/drive/MyDrive/GridNet/cache2d/upernet_convnext_t_v2 \
        --zones t1z6a,t2z5 --work /content/cache2d_work --threads 16

Resumable: a zone whose tar + manifest already exist with the right number of members and the recorded size is
skipped; interrupted conversions keep their finished local files and continue. Per zone the peak local disk use is
about twice the compact size of the zone (files + tar), at most ~30 GB for the 591-image zone t6z4. Memory: ~150 MB
per reader thread - the first version used 16 threads and a copy-heavy quantiser and was OOM-killed (exit -9) on
the 12.7 GB standard runtime; the second lost the Drive mount ("Transport endpoint is not connected", the FUSE
client died) right after a 3.4 GB tar upload while 8 threads started reading the next zone. Hence: 4 threads, a
pause after each publish, per-file read retries, and the notebook remounts Drive and re-runs on failure (finished
zones are skipped). Use a CPU High-RAM runtime.
"""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gridnet.train.maps2d import (cached_image_ids, publish_zone_tar, write_compact, write_zone_tar,  # noqa: E402
                                  zone_tar_manifest)


def zone_dirs(cache2d: str):
    return sorted(d for d in os.listdir(cache2d) if os.path.isdir(os.path.join(cache2d, d)) and "." not in d)


def compact_zone(cache2d: str, zone: str, work: str, out_dir: str, threads: int = 4, keep_work: bool = False,
                 verbose: bool = True, pause_after_publish: float = 0.0, publish: bool = True) -> dict:
    """Convert one zone and publish its tar to out_dir. With publish=False the tar + manifest are only built under
    <work>/tars/ (for an upload by other means, e.g. the Drive API) and the skip check still looks at out_dir.
    Returns the manifest (the existing one if the zone was already done)."""
    ids = cached_image_ids(cache2d, zone)
    if not ids:
        raise FileNotFoundError(f"no npz files for {zone} in {cache2d}")
    done = zone_tar_manifest(out_dir, zone, wait=30)       # a tar published minutes ago may still be uploading
    if done is not None and done["n"] == len(ids) and done.get("ids") == ids:
        if verbose:
            print(f"[compact] {zone}: {zone}.tar already complete ({done['n']} images, {done['bytes'] / 1e9:.1f} GB) - skip",
                  flush=True)
        return done
    t0 = time.time()
    local_files = os.path.join(work, "files")
    local_tars = os.path.join(work, "tars")
    todo = [i for i in ids if not os.path.exists(os.path.join(local_files, zone, i + ".npz"))]
    if verbose:
        print(f"[compact] {zone}: {len(ids)} images, {len(todo)} to convert ({threads} threads)", flush=True)
    n_done = len(ids) - len(todo)
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for _ in ex.map(lambda i: write_compact(cache2d, zone, i, local_files), todo):
            n_done += 1
            if verbose and (n_done % 50 == 0 or n_done == len(ids)):
                el = (time.time() - t0) / 60
                print(f"\r[compact] {zone} {n_done}/{len(ids)}  {el:.1f} min", end="", flush=True)
    if verbose and todo:
        print(flush=True)
    manifest = write_zone_tar(local_files, zone, ids, local_tars)
    if verbose:
        print(f"[compact] {zone}: tar built ({manifest['bytes'] / 1e9:.1f} GB, {(time.time() - t0) / 60:.1f} min)"
              + (f" -> copying to {out_dir}" if publish else f" at {local_tars}/{zone}.tar (not published)"), flush=True)
    if not keep_work:
        shutil.rmtree(os.path.join(local_files, zone), ignore_errors=True)
    if not publish:
        return manifest
    publish_zone_tar(local_tars, zone, out_dir)
    if verbose:
        print(f"[compact] {zone}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    if pause_after_publish > 0:                           # let the Drive client finish the upload before new reads
        time.sleep(pause_after_publish)
    if not keep_work:
        for suffix in (".tar", ".tar.json"):
            try:
                os.remove(os.path.join(local_tars, zone + suffix))
            except FileNotFoundError:
                pass
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache2d", required=True, help="per-image cache dir on Drive, e.g. .../cache2d/upernet_convnext_t_v2")
    ap.add_argument("--zones", default=None, help="comma list (default: every zone folder in --cache2d)")
    ap.add_argument("--out", default=None, help="where the tars go (default: --cache2d)")
    ap.add_argument("--work", default="/content/cache2d_work", help="local scratch dir")
    ap.add_argument("--threads", type=int, default=4, help="parallel Drive readers (the mount tops out at ~20-25 MB/s anyway)")
    ap.add_argument("--keep-work", action="store_true", help="keep the local compact files after publishing")
    ap.add_argument("--pause", type=float, default=60.0, help="seconds to idle after publishing a tar (Drive upload)")
    ap.add_argument("--no-publish", action="store_true",
                    help="only build <work>/tars/<zone>.tar (+ .json); the caller uploads it (Drive API path of the notebook)")
    a = ap.parse_args()
    out = a.out or a.cache2d
    zones = [z.strip() for z in a.zones.split(",") if z.strip()] if a.zones else zone_dirs(a.cache2d)
    t0 = time.time()
    total = 0
    for z in zones:
        m = compact_zone(a.cache2d, z, a.work, out, threads=a.threads, keep_work=a.keep_work, pause_after_publish=a.pause,
                         publish=not a.no_publish)
        total += m["bytes"]
    print(f"[compact] {len(zones)} zones, {total / 1e9:.1f} GB of tars in {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
