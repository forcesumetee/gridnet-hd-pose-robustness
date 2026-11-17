"""numpy-side tests for gridnet.train.data (no torch needed)."""
import numpy as np

from gridnet.train.data import CropSampler, GridIndex, Zone, cap_nearest, cover_remaining, eval_windows


def _zone(n_sparse=20_000, n_dense=60_000, seed=0) -> Zone:
    """100 x 100 m of sparse points plus a 10 x 10 m dense cluster around (50, 50)."""
    rng = np.random.default_rng(seed)
    sparse = rng.uniform(0, 100, (n_sparse, 2))
    dense = rng.uniform(45, 55, (n_dense, 2))
    xy = np.concatenate([sparse, dense]).astype(np.float32)
    xyz = np.concatenate([xy, rng.uniform(0, 5, (len(xy), 1)).astype(np.float32)], 1)
    labels = rng.integers(0, 11, len(xyz)).astype(np.uint8)
    feats = np.zeros((len(xyz), 4), np.float16)
    return Zone("synth", xyz, labels, feats, GridIndex(xyz[:, :2]), np.where(labels < 4)[0])


def test_cap_nearest_keeps_closest_points():
    z = _zone()
    c = np.array([50.0, 50.0], np.float32)
    idx = np.asarray(z.tree.query_ball_point(c, 25.0), dtype=np.int64)
    assert len(idx) > 10_000
    kept = cap_nearest(z.xyz[:, :2], idx, c, 10_000)
    assert len(kept) == 10_000
    d_kept = np.sqrt(((z.xyz[kept, :2] - c) ** 2).sum(1))
    d_all = np.sqrt(((z.xyz[idx, :2] - c) ** 2).sum(1))
    # every kept point is at least as close as every dropped point (up to ties)
    assert d_kept.max() <= np.partition(d_all, 10_000)[10_000] + 1e-6
    # small crops are untouched
    small = idx[:100]
    assert cap_nearest(z.xyz[:, :2], small, c, 10_000) is small


def test_sampler_crops_are_capped_and_contiguous():
    z = _zone()
    s = CropSampler([z], radius=25.0, max_points=5_000, rare_prob=0.0, seed=1)
    for _ in range(5):
        _, idx = s.sample()
        assert 0 < len(idx) <= 5_000


def test_eval_windows_plus_fill_in_cover_every_point():
    z = _zone()
    counts = np.zeros(z.n, np.int32)
    n_reg = 0
    for idx in eval_windows(z, radius=25.0, max_points=8_000):
        assert len(idx) <= 8_000
        counts[idx] += 1
        n_reg += 1
    assert (counts == 0).sum() > 0, "the dense cluster should exceed the cap and leave gaps"
    n_extra = 0
    for idx in cover_remaining(z, counts, radius=25.0, max_points=8_000):
        assert len(idx) <= 8_000
        counts[idx] += 1
        n_extra += 1
    assert (counts > 0).all()
    assert n_extra > 0
