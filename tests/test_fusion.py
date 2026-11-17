import json
import os

import numpy as np
import pytest
torch = pytest.importorskip("torch")   # GPU-side tests: skipped on the torch-less Mac venv
import torch.nn as nn

from gridnet.camera import Calibration, Pose, rot_from_opk
from gridnet.train.assoc_gpu import associate, depth_maps, project_t, zone_cams
from gridnet.train.engine_fusion import FrozenBackbone, FusionTrainer, load_zone_context, label_counts, class_weights_from_counts
from gridnet.train.fusion import (ConcatHead, LateFusionHead, WindowAttnHead, ZoneMaps, assoc_rows, lift, perturb_cams,
                                  project_pairs_t, reproject_rows, uniform_weights)

CAL = Calibration(width=400, height=300, f=180.0, k1=-0.05, k2=0.01, p1=1e-4, p2=-1e-4)
ORIGIN = np.array([500000.0, 5100000.0, 400.0])


def make_poses():
    poses = {}
    for i, (dx, dy, kappa) in enumerate([(-30, 0, 0), (30, 10, 90), (0, -40, 180), (10, 40, 45)]):
        R = rot_from_opk(0.0, 0.0, kappa)
        poses[f"img{i}"] = Pose(S=ORIGIN + np.array([dx, dy, 60.0]), R=R, opk=np.array([0, 0, kappa]))
    return poses


def write_zone(root, zone="zz", n=12000, seed=0):
    """LiDAR cache + camera files + compact 2D cache for a synthetic zone. Returns (cache_dir, cache2d_dir, data_dir)."""
    rng = np.random.default_rng(seed)
    xyz = np.column_stack([rng.uniform(-60, 60, n), rng.uniform(-50, 50, n), rng.uniform(0, 1, n)]).astype(np.float32)
    xyz[: n // 10, 2] += rng.uniform(5, 20, n // 10).astype(np.float32)
    labels = rng.choice([0, 1, 4, 6, 8, 255], size=n, p=[0.05, 0.05, 0.3, 0.45, 0.1, 0.05]).astype(np.uint8)
    cache = os.path.join(root, "cache"); os.makedirs(os.path.join(cache, zone))
    geo = {k: rng.random(n).astype(np.float16) for k in ("linearity", "planarity", "sphericity", "verticality", "hag")}
    nrm = np.zeros((n, 3), np.float16); nrm[:, 2] = 1
    np.savez(os.path.join(cache, zone, "points.npz"), xyz=xyz, origin=ORIGIN, labels=labels, normals=nrm, **geo)
    data = os.path.join(root, "data"); os.makedirs(os.path.join(data, zone, "pose"))
    open(os.path.join(data, zone, "pose", "camera_calibration.xml"), "w").write(
        f"<calibration><width>{CAL.width}</width><height>{CAL.height}</height><f>{CAL.f}</f><k1>{CAL.k1}</k1>"
        f"<k2>{CAL.k2}</k2><p1>{CAL.p1}</p1><p2>{CAL.p2}</p2></calibration>")
    poses = make_poses()
    lines = ["# PhotoID X Y Z Omega Phi Kappa r11..r33"]
    for pid, p in poses.items():
        lines.append(pid + " " + " ".join(f"{v:.9f}" for v in [*p.S, *p.opk, *p.R.ravel()]))
    open(os.path.join(data, zone, "pose", "cameras_pose.txt"), "w").write("\n".join(lines) + "\n")
    cache2d = os.path.join(root, "cache2d"); os.makedirs(os.path.join(cache2d, zone))
    H4, W4 = CAL.height // 4, CAL.width // 4
    H8, W8 = H4 // 2, W4 // 2
    for j, pid in enumerate(sorted(poses)):
        prob = np.zeros((11, H4, W4), np.uint8); prob[j % 11] = 255          # image j votes class j
        feat = rng.integers(-127, 128, (32, H8, W8), dtype=np.int8)
        np.savez(os.path.join(cache2d, zone, pid + ".npz"), prob=prob, feat=feat, feat_scale=np.full(32, 0.01, np.float32))
    return cache, cache2d, data


class DummySeg(nn.Module):
    """Stands in for build_model(): .backbone(batch).feat (N,72) and .head(feat) (N,11)."""

    class _Out:
        def __init__(self, feat):
            self.feat = feat

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__(); self.lin = nn.Linear(7, 72)

        def forward(self, batch):
            return DummySeg._Out(self.lin(batch["feat"]))

    def __init__(self):
        super().__init__()
        self.backbone = self._Backbone(); self.head = nn.Linear(72, 11)


def test_zone_maps_gathers_match_numpy(tmp_path):
    cache, cache2d, data = write_zone(str(tmp_path))
    maps = ZoneMaps(cache2d, "zz", ["img0", "img1", "img2", "img3"], device="cpu", verbose=False)
    img = torch.tensor([1, 3, 0]); u = torch.tensor([17.0, 399.0, 0.0]); v = torch.tensor([9.0, 299.0, 0.0])
    p = maps.gather_prob(img, u, v)
    assert p.shape == (3, 11) and p[0, 1] == 1.0 and p[1, 3] == 1.0 and p[2, 0] == 1.0
    f = maps.gather_feat(img, u, v)
    raw = np.load(os.path.join(cache2d, "zz", "img1.npz"))
    assert np.isclose(f[0, 5].item(), raw["feat"][5, 9 // 8, 17 // 8] * 0.01, atol=1e-6)
    w = maps.gather_window(img, u, v, 3)
    assert w.shape == (3, 9, 32)
    assert torch.allclose(w[0, 4], f[0])                         # centre cell of the window = the centre gather
    assert torch.allclose(w[1, 8], w[1, 4])                      # bottom-right corner clamps to the image border


def test_heads_shapes_and_window_k1_matches_concat():
    N, V = 50, 8
    lidar_logits, lidar_feat = torch.randn(N, 11), torch.randn(N, 72)
    img_prob, img_feat = torch.rand(N, V, 11), torch.randn(N, V, 32)
    desc = torch.randn(N, V, 5); mask = torch.rand(N, V) > 0.3; mask[0] = False
    assert LateFusionHead()(lidar_logits, lidar_feat, img_prob, None, desc, mask).shape == (N, 11)
    assert ConcatHead()(lidar_logits, lidar_feat, img_prob, img_feat, desc, mask).shape == (N, 11)
    w = WindowAttnHead(k=3)
    assert w(lidar_logits, lidar_feat, img_prob, torch.randn(N, V, 9, 32), desc, mask).shape == (N, 11)
    assert w.last_attention.shape == (N, V, 9)
    assert torch.allclose(uniform_weights(mask)[1].sum(), torch.tensor(1.0)) and uniform_weights(mask)[0].sum() == 0


def test_pose_perturbation_and_reprojection(tmp_path):
    cache, cache2d, data = write_zone(str(tmp_path))
    ctx = load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), "zz", k=3, need_depths=True, device="cpu", verbose=False)
    assert os.path.exists(tmp_path / "assoc" / "zz" / "assoc.npz") and ctx.depths is not None
    S2, R2 = perturb_cams(ctx.S, ctx.R, 0.0, 0.0)
    assert torch.allclose(S2, ctx.S) and torch.allclose(R2, ctx.R)
    S3, R3 = perturb_cams(ctx.S, ctx.R, 0.1, 0.1, torch.Generator().manual_seed(0))
    assert (S3 - ctx.S).norm(dim=1).mean() > 0.05 and torch.allclose(R3 @ R3.transpose(1, 2), torch.eye(3).expand(4, 3, 3), atol=1e-5)
    # per-pair projection equals the per-camera projection
    idx = np.arange(0, 2000)
    rows = assoc_rows(ctx.assoc, idx, "cpu")
    xyz = torch.from_numpy(ctx.zone.xyz[idx])
    flat = rows["img"].reshape(-1); pts = xyz[:, None, :].expand(-1, 3, 3).reshape(-1, 3)
    u, v, z, ok = project_pairs_t(pts, ctx.S[flat], ctx.R[flat], ctx.cal)
    u0, v0, z0, ok0 = project_t(pts[flat == 2], ctx.S[2], ctx.R[2], ctx.cal)
    assert torch.allclose(u[flat == 2], u0, atol=1e-3) and torch.allclose(z[flat == 2], z0, atol=1e-4)
    # zero noise: (u, v) of the stored association are reproduced and every stored view stays visible
    rr = reproject_rows(rows, xyz, ctx.S, ctx.R, ctx.cal, ctx.depths, ctx.W, tau=0.5)
    m = rows["mask"]
    assert (rr["mask"] == m).float().mean() > 0.99
    assert (rr["u"][m] - rows["u"][m]).abs().max() < 1.0 and (rr["v"][m] - rows["v"][m]).abs().max() < 1.0
    # large noise moves pixels and drops some pairs
    S4, R4 = perturb_cams(ctx.S, ctx.R, 5.0, 2.0, torch.Generator().manual_seed(1))
    r4 = reproject_rows(rows, xyz, S4, R4, ctx.cal, ctx.depths, ctx.W)
    assert (r4["u"][r4["mask"]] - rows["u"][r4["mask"]]).abs().mean() > 1.0


def test_trainer_fit_eval_resume_all_heads(tmp_path):
    cache, cache2d, data = write_zone(str(tmp_path))
    backbone = FrozenBackbone("", "", model=DummySeg(), device="cpu")
    cw = class_weights_from_counts(label_counts(cache, ["zz"]))
    assert cw.shape == (11,) and cw[0] > cw[6]
    for head_name, head in [("late", LateFusionHead()), ("concat", ConcatHead()), ("window", WindowAttnHead(k=3))]:
        cfg = {"head": head_name, "train_zones": ["zz"], "val_zones": ["zz"], "passes": 1, "its_per_zone": 3, "seed": 0,
               "radius": 25.0, "max_points": 3000, "eval_max_points": 4000, "grid": 0.05, "lr": 1e-3, "head_points": 1500,
               "head_chunk": 700, "log_every": 1, "ckpt_every": 2, "eval_every_pass": True,
               "pose_noise_t": 0.1 if head_name == "concat" else 0.0, "pose_noise_r": 0.1 if head_name == "concat" else 0.0, "aug_prob": 1.0}
        work = str(tmp_path / f"work_{head_name}")
        tr = FusionTrainer(backbone, head, cfg, work, class_w=cw, device="cpu")
        loader = lambda zone, need_depths: load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), zone, k=3,
                                                              need_depths=need_depths, with_feat=head.uses_feat,
                                                              device="cpu", verbose=False)
        status = tr.fit(loader, ["zz"], eval_fn=lambda t: t.evaluate(loader, ["zz"], save_dir=os.path.join(work, "eval"))["mIoU"])
        assert status == "done" and tr.it == 3 and tr.best > -1
        e = json.load(open(os.path.join(work, "eval", "eval.json")))
        assert set(e) >= {"mIoU", "OA", "per_class", "per_zone", "confusion", "lidar_only"}
        pred = np.load(os.path.join(work, "zz" if False else "eval", "zz", "fused_pred.npy"))
        assert pred.shape == (12000,) and pred.max() < 11
        # resume into a fresh trainer continues from the cursor
        tr2 = FusionTrainer(backbone, type(head)(k=3) if head_name == "window" else type(head)(), cfg, work, class_w=cw, device="cpu")
        assert tr2.resume("last.pth") and tr2.it == 3 and tr2.cursor == {"pass": 1, "zone": 0, "done": 0}
        assert tr2.fit(loader, ["zz"]) == "done" and tr2.it == 3   # nothing left to do


def test_time_limit_saves_cursor(tmp_path):
    cache, cache2d, data = write_zone(str(tmp_path))
    backbone = FrozenBackbone("", "", model=DummySeg(), device="cpu")
    cfg = {"head": "late", "train_zones": ["zz"], "val_zones": ["zz"], "passes": 2, "its_per_zone": 4, "seed": 1, "radius": 25.0,
           "max_points": 3000, "grid": 0.05, "lr": 1e-3, "head_points": 1000, "head_chunk": 500, "log_every": 100, "ckpt_every": 100}
    tr = FusionTrainer(backbone, LateFusionHead(), cfg, str(tmp_path / "w"), device="cpu")
    loader = lambda zone, nd: load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), zone, k=3, with_feat=False,
                                                device="cpu", verbose=False)
    assert tr.fit(loader, ["zz"], max_hours=0.0) == "time limit" and tr.it == 1 and tr.cursor == {"pass": 0, "zone": 0, "done": 1}
    tr2 = FusionTrainer(backbone, LateFusionHead(), cfg, str(tmp_path / "w"), device="cpu")
    assert tr2.resume("last.pth") and tr2.fit(loader, ["zz"]) == "done" and tr2.it == 8


def test_best_eval_is_not_recomputed_and_assoc_source(tmp_path):
    cache, cache2d, data = write_zone(str(tmp_path))
    backbone = FrozenBackbone("", "", model=DummySeg(), device="cpu")
    cfg = {"head": "late", "train_zones": ["zz"], "val_zones": ["zz"], "passes": 2, "its_per_zone": 2, "seed": 0, "radius": 25.0,
           "max_points": 3000, "eval_max_points": 4000, "grid": 0.05, "lr": 1e-3, "head_points": 1000, "head_chunk": 500,
           "log_every": 100, "ckpt_every": 100, "eval_every_pass": True}
    work = str(tmp_path / "w"); save_dir = os.path.join(work, "eval"); adir = str(tmp_path / "assoc")
    loads = []

    def loader(zone, need_depths, source="auto"):
        loads.append(source)
        return load_zone_context(cache, cache2d, data, adir, zone, k=3, with_feat=False, device="cpu", verbose=False,
                                 assoc_source=source)
    tr = FusionTrainer(backbone, LateFusionHead(), cfg, work, device="cpu")
    scores = iter([50.0, 40.0])                                   # pass 1 is the best, pass 2 is worse

    def eval_fn(t):
        t.evaluate(loader, ["zz"], save_dir=save_dir, tag="pass")
        return next(scores)
    assert tr.fit(loader, ["zz"], eval_fn=eval_fn) == "done" and tr.best == 50.0
    assert json.load(open(os.path.join(save_dir, "eval.json")))["it"] == 4        # eval/ holds pass 2 (it 4) ...
    n = len(loads)
    summ, redone = tr.ensure_best_eval(loader, ["zz"], save_dir)                    # ... so the best (it 2) is re-evaluated
    assert redone and summ["tag"] == "best" and summ["it"] == 2 and len(loads) == n + 1
    summ2, redone2 = tr.ensure_best_eval(loader, ["zz"], save_dir)                  # now it is there: nothing to do
    assert not redone2 and summ2["it"] == 2 and len(loads) == n + 1
    # when the last pass is the best one, the pass evaluation is simply re-tagged
    work_b = str(tmp_path / "wb"); save_b = os.path.join(work_b, "eval")
    trb = FusionTrainer(backbone, LateFusionHead(), cfg, work_b, device="cpu")
    scores_b = iter([40.0, 50.0])
    trb.fit(loader, ["zz"], eval_fn=lambda t: (t.evaluate(loader, ["zz"], save_dir=save_b, tag="pass"), next(scores_b))[1])
    n = len(loads)
    summ3, redone3 = trb.ensure_best_eval(loader, ["zz"], save_b)
    assert not redone3 and summ3["tag"] == "best" and summ3["it"] == 4 and len(loads) == n
    assert json.load(open(os.path.join(save_b, "eval.json")))["tag"] == "best"

    # a run whose final weights were never validated (session died in the last evaluation): finish() validates them
    work_c = str(tmp_path / "wc"); save_c = os.path.join(work_c, "eval")
    trc = FusionTrainer(backbone, LateFusionHead(), cfg, work_c, device="cpu")
    assert trc.fit(loader, ["zz"]) == "done" and trc.last_eval_it == -1 and not os.path.exists(os.path.join(work_c, "best.pth"))
    summ4, redone4 = trc.finish(loader, ["zz"], save_c)
    assert not redone4 and summ4["tag"] == "best" and summ4["it"] == 4 and os.path.exists(os.path.join(work_c, "best.pth"))
    trd = FusionTrainer(backbone, LateFusionHead(), cfg, work_c, device="cpu")       # resume keeps that knowledge
    assert trd.resume("last.pth") and trd.last_eval_it == 4
    n = len(loads); trd.finish(loader, ["zz"], save_c); assert len(loads) == n       # nothing recomputed

    # association source: the archive is written once and never overwritten; "compute" rebuilds the same table
    apath = os.path.join(adir, "zz", "assoc.npz")
    assert os.path.exists(apath)
    stamp = os.path.getmtime(apath)
    a_arch = loader("zz", False, "archive").assoc
    a_comp = loader("zz", False, "compute").assoc
    assert os.path.getmtime(apath) == stamp
    assert np.array_equal(a_arch.img, a_comp.img) and np.array_equal(a_arch.u, a_comp.u) and np.array_equal(a_arch.n_views, a_comp.n_views)
    with pytest.raises(AssertionError):
        loader("zz", False, "drive")


def test_validation_is_repeatable_with_a_stochastic_backbone(tmp_path):
    """LitePT shuffles its serialization orders with torch.randperm in every forward pass; predict_zone must give
    the same answer twice and must not disturb the global random stream used for training."""
    from gridnet.train.engine_fusion import fixed_torch_seed

    class NoisySeg(DummySeg):
        class _Backbone(DummySeg._Backbone):
            def forward(self, batch):
                out = super().forward(batch)
                out.feat = out.feat + 0.5 * torch.randn(out.feat.shape)        # stands in for the random order
                return out

        def __init__(self):
            nn.Module.__init__(self)
            self.backbone = self._Backbone(); self.head = nn.Linear(72, 11)

    cache, cache2d, data = write_zone(str(tmp_path))
    torch.manual_seed(0)
    backbone = FrozenBackbone("", "", model=NoisySeg(), device="cpu")
    cfg = {"head": "late", "train_zones": ["zz"], "val_zones": ["zz"], "passes": 1, "its_per_zone": 1, "seed": 0, "radius": 25.0,
           "max_points": 3000, "eval_max_points": 4000, "grid": 0.05, "lr": 1e-3, "head_points": 1000, "head_chunk": 500}
    tr = FusionTrainer(backbone, LateFusionHead(), cfg, str(tmp_path / "w"), device="cpu")
    ctx = load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), "zz", k=3, with_feat=False, device="cpu", verbose=False)
    f1, l1 = tr.predict_zone(ctx)
    f2, l2 = tr.predict_zone(ctx)
    assert np.array_equal(f1, f2) and np.array_equal(l1, l2)
    with fixed_torch_seed(1):                                            # another seed -> another realisation
        _, l3 = tr._predict_zone(ctx)
    assert not np.array_equal(l1, l3)
    torch.manual_seed(7); a = torch.rand(3)
    torch.manual_seed(7); tr.predict_zone(ctx); b = torch.rand(3)        # the surrounding stream is untouched
    assert torch.equal(a, b)


def test_parameter_counts_of_the_cost_table_are_those_of_the_heads():
    """results/compute_cost.json feeds a table of the paper: its head sizes must be the sizes of the modules."""
    from gridnet.train.fusion import build_head
    cost = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "compute_cost.json")))
    rows = {r["component"]: r["params"] for r in cost["rows"] if "component" in r}
    n = lambda name, **kw: sum(p.numel() for p in build_head(name, **kw).parameters())
    assert rows["Late fusion head"] == n("late") and rows["Feature concatenation head"] == n("concat")
    assert rows["Window head, $3{\\times}3$ / $5{\\times}5$"] == [n("window", k=3), n("window", k=5)]

