"""Readers for the per-image 2D cache written by `Trainer2D.export_cache`.

Two on-disk formats exist: fp16 (`prob` (11,H/4,W/4) float16, `feat` (32,H/8,W/8) float16; the first val-zone
caches, ~50 MB/image) and compact (`prob` uint8 = round(255 p), `feat` int8 with a per-channel `feat_scale`
(32,) float32 so that feat ~= int8 * scale; ~24 MB/image). `load_maps` returns the compact representation for
both, which is also what `ZoneMaps` keeps resident on the GPU: uint8 / int8 maps + float32 scales.
"""
import os
from typing import Optional, Sequence, Tuple

import numpy as np

PROB_SCALE = 255.0


def quantize_feat(feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """fp16/fp32 (C,H,W) -> (int8 (C,H,W), float32 scale (C,)) with feat ~= int8 * scale.
    In-place arithmetic on one float32 copy, so the peak is ~1.5x the fp32 size (16 threads of this run on a
    12.7 GB Colab CPU runtime)."""
    f = feat.astype(np.float32)
    scale = np.maximum(np.abs(f).max(axis=(1, 2)), 1e-6) / 127.0
    f /= scale[:, None, None]
    np.round(f, out=f)
    np.clip(f, -127, 127, out=f)
    return f.astype(np.int8), scale.astype(np.float32)


def quantize_prob(prob: np.ndarray) -> np.ndarray:
    """fp16/fp32 probabilities -> uint8 (x255), in place on one float32 copy."""
    q = prob.astype(np.float32)
    q *= PROB_SCALE
    np.round(q, out=q)
    np.clip(q, 0, 255, out=q)
    return q.astype(np.uint8)


def load_maps(cache2d_dir: str, zone: str, image_id: str, what: Sequence[str] = ("prob", "feat")):
    """Returns a dict with any of: prob (uint8, (11,H/4,W/4)), feat (int8, (32,H/8,W/8)), feat_scale (float32,(32,)).
    fp16 caches are quantised on the fly so that callers see one format."""
    out = {}
    with np.load(os.path.join(cache2d_dir, zone, image_id + ".npz")) as d:
        if "prob" in what:
            p = d["prob"]
            out["prob"] = p if p.dtype == np.uint8 else quantize_prob(p)
        if "feat" in what:
            f = d["feat"]
            if f.dtype == np.int8:
                out["feat"], out["feat_scale"] = f, d["feat_scale"].astype(np.float32)
            else:
                out["feat"], out["feat_scale"] = quantize_feat(f)
    return out


def load_prob_fp16(cache2d_dir: str, zone: str, image_id: str) -> np.ndarray:
    """(11, H/4, W/4) float16 probabilities (dequantised if the cache is compact)."""
    with np.load(os.path.join(cache2d_dir, zone, image_id + ".npz")) as d:
        p = d["prob"]
    return p if p.dtype == np.float16 else (p.astype(np.float32) / PROB_SCALE).astype(np.float16)


def cached_image_ids(cache2d_dir: str, zone: str) -> list:
    zdir = os.path.join(cache2d_dir, zone)
    if not os.path.isdir(zdir):
        return []
    return sorted(f[:-4] for f in os.listdir(zdir) if f.endswith(".npz") and not f.endswith(".tmp.npz"))


# ---------------------------------------------------------------------------------------------------------------
# Zone tars.  Colab reads Drive through FUSE at ~10-15 MB/s when it is one npz file after another, but at
# >100 MB/s for one big sequential file, so a zone's cache is also kept on Drive as ONE uncompressed tar of
# compact npz members (`<cache2d>/<zone>.tar`, members `<zone>/<image>.npz`) next to a small manifest
# `<zone>.tar.json` {"zone", "n", "ids", "bytes"}.  `scripts/compact_cache2d.py` writes them once; the training
# notebooks stream-extract them to local disk instead of copying hundreds of files.
# ---------------------------------------------------------------------------------------------------------------

def write_compact(cache2d_dir: str, zone: str, image_id: str, dst_dir: str) -> str:
    """Re-write one cached image in the compact format under <dst_dir>/<zone>/<image_id>.npz (fp16 sources are
    quantised exactly as `load_maps` does, compact sources are copied unchanged). Resumable: existing files are
    kept. Returns the destination path."""
    import time
    zdir = os.path.join(dst_dir, zone)
    os.makedirs(zdir, exist_ok=True)
    dst = os.path.join(zdir, image_id + ".npz")
    if os.path.exists(dst):
        return dst
    for attempt in range(3):                            # Colab's Drive mount has transient read errors
        try:
            m = load_maps(cache2d_dir, zone, image_id)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5)
    tmp = dst + ".tmp.npz"
    np.savez(tmp, prob=m["prob"], feat=m["feat"], feat_scale=m["feat_scale"])
    os.replace(tmp, dst)
    return dst


def write_zone_tar(local_dir: str, zone: str, image_ids: Sequence[str], out_dir: str) -> dict:
    """Pack <local_dir>/<zone>/<id>.npz for the given ids into <out_dir>/<zone>.tar (uncompressed, members
    '<zone>/<id>.npz') plus the manifest <zone>.tar.json. The tar is built next to the destination under a temporary
    name and renamed at the end, so a half-written tar is never mistaken for a finished one. Returns the manifest."""
    import json
    import tarfile
    os.makedirs(out_dir, exist_ok=True)
    tar_path = os.path.join(out_dir, zone + ".tar")
    tmp_path = tar_path + ".tmp"
    with tarfile.open(tmp_path, "w") as tf:
        for iid in sorted(image_ids):
            tf.add(os.path.join(local_dir, zone, iid + ".npz"), arcname=f"{zone}/{iid}.npz", recursive=False)
    manifest = {"zone": zone, "n": len(image_ids), "ids": sorted(image_ids), "bytes": os.path.getsize(tmp_path),
                "format": "compact"}
    os.replace(tmp_path, tar_path)
    with open(tar_path + ".json.tmp", "w") as fh:
        json.dump(manifest, fh)
    os.replace(tar_path + ".json.tmp", tar_path + ".json")
    return manifest


def publish_zone_tar(src_dir: str, zone: str, dst_dir: str) -> dict:
    """Copy <src_dir>/<zone>.tar (+ manifest) built on local disk to <dst_dir> (Drive) with one large sequential
    copy; the tar lands under a temporary name and is renamed once complete, the manifest last."""
    import json
    import shutil
    os.makedirs(dst_dir, exist_ok=True)
    src_tar = os.path.join(src_dir, zone + ".tar")
    with open(src_tar + ".json") as fh:
        manifest = json.load(fh)
    dst_tar = os.path.join(dst_dir, zone + ".tar")
    shutil.copyfile(src_tar, dst_tar + ".tmp")
    assert os.path.getsize(dst_tar + ".tmp") == manifest["bytes"], f"short copy of {zone}.tar"
    os.replace(dst_tar + ".tmp", dst_tar)
    with open(dst_tar + ".json.tmp", "w") as fh:
        json.dump(manifest, fh)
    os.replace(dst_tar + ".json.tmp", dst_tar + ".json")
    return manifest


def zone_tar_manifest(cache2d_dir: str, zone: str, wait: float = 0.0) -> Optional[dict]:
    """The manifest of <cache2d_dir>/<zone>.tar if the tar exists and has the size the manifest records, else None.
    On Colab's Drive mount a tar written moments ago can report a smaller size until its upload has finished, so
    `wait` (seconds) re-checks every 5 s before giving up."""
    import json
    import time
    tar_path = os.path.join(cache2d_dir, zone + ".tar")
    man_path = tar_path + ".json"
    deadline = time.time() + wait
    while True:
        if not (os.path.exists(tar_path) and os.path.exists(man_path)):
            return None                                     # absent: no point waiting
        with open(man_path) as fh:
            manifest = json.load(fh)
        if os.path.getsize(tar_path) == manifest.get("bytes"):
            return manifest
        if time.time() >= deadline:                         # present but short: upload in flight or truncated
            return None
        time.sleep(5)


def extract_zone_tar(cache2d_dir: str, zone: str, dst_root: str) -> int:
    """Stream-extract <cache2d_dir>/<zone>.tar into <dst_root>/<zone>/ (one sequential pass, no second copy of the
    tar on disk; tarfile's default block size - a large `bufsize` makes stream mode quadratic). Skipped when the
    destination already holds every image of the manifest. Returns the number of images available locally."""
    import tarfile
    manifest = zone_tar_manifest(cache2d_dir, zone)
    if manifest is None:
        raise FileNotFoundError(f"no complete {zone}.tar in {cache2d_dir}")
    have = set(cached_image_ids(dst_root, zone))
    if have.issuperset(manifest["ids"]):
        return len(have)
    os.makedirs(dst_root, exist_ok=True)
    with tarfile.open(os.path.join(cache2d_dir, zone + ".tar"), "r|") as tf:
        try:
            tf.extractall(dst_root, filter="data")
        except TypeError:                                   # Python < 3.10.12 without the `filter` argument
            tf.extractall(dst_root)
    return len(cached_image_ids(dst_root, zone))
