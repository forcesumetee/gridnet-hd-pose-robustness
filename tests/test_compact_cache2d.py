"""numpy-only tests for the zone tars of the 2D cache (gridnet.train.maps2d + scripts/compact_cache2d.py)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from compact_cache2d import compact_zone, zone_dirs  # noqa: E402
from gridnet.train.maps2d import (cached_image_ids, extract_zone_tar, load_maps, zone_tar_manifest)  # noqa: E402


def write_fp16_cache(root, zone, ids, seed=0, compact=False):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, zone), exist_ok=True)
    for iid in ids:
        prob = rng.dirichlet(np.ones(11), size=(12, 16)).transpose(2, 0, 1).astype(np.float16)   # (11, 12, 16)
        feat = (rng.normal(size=(32, 6, 8)) * rng.uniform(0.1, 3.0, size=(32, 1, 1))).astype(np.float16)
        if compact:
            from gridnet.train.maps2d import quantize_feat
            q, s = quantize_feat(feat)
            np.savez(os.path.join(root, zone, iid + ".npz"), prob=np.round(prob.astype(np.float32) * 255).astype(np.uint8),
                     feat=q, feat_scale=s)
        else:
            np.savez(os.path.join(root, zone, iid + ".npz"), prob=prob, feat=feat)


def test_zone_tar_round_trip_and_resume(tmp_path):
    cache = tmp_path / "cache2d"; work = tmp_path / "work"; local = tmp_path / "local"
    ids = ["img_b", "img_a", "img_c"]
    write_fp16_cache(str(cache), "zz1", ids)
    write_fp16_cache(str(cache), "zz2", ["only"], seed=1, compact=True)
    assert zone_dirs(str(cache)) == ["zz1", "zz2"]

    m = compact_zone(str(cache), "zz1", str(work), str(cache), threads=2, verbose=False)
    assert m["n"] == 3 and m["ids"] == sorted(ids)
    assert os.path.exists(cache / "zz1.tar") and os.path.exists(cache / "zz1.tar.json")
    assert zone_tar_manifest(str(cache), "zz1") == m
    assert not os.path.exists(work / "files" / "zz1")              # scratch cleaned up
    # a second call is a no-op (same manifest object contents)
    m2 = compact_zone(str(cache), "zz1", str(work), str(cache), threads=2, verbose=False)
    assert m2 == m

    # stream extraction gives back exactly what load_maps returns for the fp16 source
    n = extract_zone_tar(str(cache), "zz1", str(local))
    assert n == 3 and cached_image_ids(str(local), "zz1") == sorted(ids)
    for iid in ids:
        a, b = load_maps(str(cache), "zz1", iid), load_maps(str(local), "zz1", iid)
        assert a["prob"].dtype == np.uint8 and b["prob"].dtype == np.uint8
        assert np.array_equal(a["prob"], b["prob"]) and np.array_equal(a["feat"], b["feat"])
        assert np.allclose(a["feat_scale"], b["feat_scale"])
        with np.load(local / "zz1" / (iid + ".npz")) as d:          # on-disk format is compact
            assert d["prob"].dtype == np.uint8 and d["feat"].dtype == np.int8 and d["feat_scale"].shape == (32,)
    # extracting again is skipped (everything already present)
    assert extract_zone_tar(str(cache), "zz1", str(local)) == 3

    # a compact source is packed unchanged
    compact_zone(str(cache), "zz2", str(work), str(cache), threads=1, verbose=False)
    extract_zone_tar(str(cache), "zz2", str(local))
    with np.load(cache / "zz2" / "only.npz") as src, np.load(local / "zz2" / "only.npz") as dst:
        assert np.array_equal(src["prob"], dst["prob"]) and np.array_equal(src["feat"], dst["feat"])


def test_incomplete_tar_is_not_trusted(tmp_path):
    cache = tmp_path / "cache2d"
    write_fp16_cache(str(cache), "zz", ["a", "b"])
    compact_zone(str(cache), "zz", str(tmp_path / "work"), str(cache), threads=1, verbose=False)
    man = json.load(open(cache / "zz.tar.json"))
    with open(cache / "zz.tar", "ab") as fh:                        # size no longer matches the manifest
        fh.write(b"\0" * 512)
    assert zone_tar_manifest(str(cache), "zz") is None
    # a new image in the folder makes the manifest stale -> the zone is rebuilt
    with open(cache / "zz.tar", "r+b") as fh:
        fh.truncate(man["bytes"])
    assert zone_tar_manifest(str(cache), "zz") is not None
    write_fp16_cache(str(cache), "zz", ["c"], seed=3)
    m = compact_zone(str(cache), "zz", str(tmp_path / "work"), str(cache), threads=1, verbose=False)
    assert m["n"] == 3 and zone_tar_manifest(str(cache), "zz")["ids"] == ["a", "b", "c"]
