import os

import numpy as np
import pytest
torch = pytest.importorskip("torch")   # GPU-side tests: skipped on the torch-less Mac venv

from gridnet.camera import Calibration, Pose, rot_from_opk
from gridnet.preprocess.association import Association, build_association
from gridnet.train.assoc_gpu import associate, project_t
from gridnet.camera import project
from gridnet.train.imagevote import vote


def scene(seed=0, n=20000):
    rng = np.random.default_rng(seed)
    xyz = np.column_stack([rng.uniform(-100, 100, n), rng.uniform(-80, 80, n), rng.uniform(0, 2, n)])
    xyz[: n // 10, 2] += rng.uniform(5, 25, n // 10)              # some elevated points that occlude ground
    xyz += np.array([500000.0, 5100000.0, 400.0])                  # UTM-like absolute coordinates
    cal = Calibration(width=400, height=300, f=180.0, k1=-0.05, k2=0.01, p1=1e-4, p2=-1e-4)
    poses = {}
    for i, (dx, dy, kappa) in enumerate([(-30, 0, 0), (30, 10, 90), (0, -40, 180), (10, 40, 45)]):
        R = rot_from_opk(0.0, 0.0, kappa)
        poses[f"img{i}"] = Pose(S=np.array([500000.0 + dx, 5100000.0 + dy, 460.0]), R=R, opk=np.array([0, 0, kappa]))
    return xyz, cal, poses


def test_project_t_matches_numpy():
    xyz, cal, poses = scene()
    pose = poses["img0"]
    u, v, z, ok = project(xyz, pose, cal)
    origin = xyz.min(0)
    S = torch.as_tensor(pose.S - origin, dtype=torch.float32); R = torch.as_tensor(pose.R, dtype=torch.float32)
    ut, vt, zt, okt = project_t(torch.as_tensor(xyz - origin, dtype=torch.float32), S, R, cal)
    assert (okt.numpy() == ok).mean() > 0.999
    m = ok & okt.numpy()
    assert np.abs(ut.numpy()[m] - u[m]).max() < 0.05 and np.abs(vt.numpy()[m] - v[m]).max() < 0.05
    assert np.abs(zt.numpy()[m] - z[m]).max() < 1e-3


def test_associate_matches_numpy_reference():
    xyz, cal, poses = scene()
    ref = build_association(xyz, poses, cal, k=3, verbose=False)
    origin = xyz.min(0)
    got = associate((xyz - origin).astype(np.float32), origin, poses, cal, k=3, chunk=7000, device="cpu", verbose=False)
    assert got.image_ids == ref.image_ids
    assert (got.n_views == ref.n_views).mean() > 0.995
    same_sets = np.mean([set(a[a >= 0].tolist()) == set(b[b >= 0].tolist()) for a, b in zip(got.img, ref.img)])
    assert same_sets > 0.99
    both = (got.img >= 0) & (ref.img >= 0) & (got.img == ref.img)
    assert np.abs(got.u[both].astype(int) - ref.u[both].astype(int)).max() <= 1
    assert np.abs(got.v[both].astype(int) - ref.v[both].astype(int)).max() <= 1
    assert np.allclose(got.feat[both][:, 0].astype(float), ref.feat[both][:, 0].astype(float), atol=0.1)  # distance


def test_vote_reads_the_cached_probabilities(tmp_path):
    xyz, cal, poses = scene(n=3000)
    origin = xyz.min(0)
    assoc = associate((xyz - origin).astype(np.float32), origin, poses, cal, k=2, device="cpu", verbose=False)
    zone = "zz"; os.makedirs(tmp_path / zone)
    H4, W4 = cal.height // 4, cal.width // 4
    for j, iid in enumerate(assoc.image_ids):          # image j votes class j everywhere
        prob = np.zeros((11, H4, W4), np.float16); prob[j] = 1.0
        np.savez(tmp_path / zone / f"{iid}.npz", prob=prob, feat=np.zeros((1, 1, 1), np.float16))
    prob, counts = vote(assoc, str(tmp_path), zone, device="cpu", verbose=False)
    assert prob.shape == (len(xyz), 11) and (counts == (assoc.img >= 0).sum(1)).all()
    seen = counts > 0
    assert np.allclose(prob[seen].sum(1), 1.0, atol=1e-5)
    # the voted class of a point with a single view is that view's image index
    single = counts == 1
    if single.any():
        first = np.where(assoc.img[single] >= 0, assoc.img[single], 99).min(1)
        assert (prob[single].argmax(1) == first).all()
