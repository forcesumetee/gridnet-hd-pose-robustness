import json
import os
import sys

import numpy as np
import pytest
torch = pytest.importorskip("torch")   # GPU-side tests: skipped on the torch-less Mac venv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fusion import DummySeg, write_zone  # noqa: E402

from gridnet.train.engine_fusion import FrozenBackbone, load_zone_context  # noqa: E402
from gridnet.train.fusion import ConcatHead, LateFusionHead, WindowAttnHead  # noqa: E402
from gridnet.train import protocol as P  # noqa: E402


def test_stratified_subset_weights_reproduce_class_totals():
    rng = np.random.default_rng(0)
    labels = rng.choice([0, 1, 4, 6, 255], size=50_000, p=[0.02, 0.03, 0.4, 0.5, 0.05]).astype(np.uint8)
    idx, w = P.stratified_subset(labels, rng, cap=5_000, cap_full=1_000)
    assert np.all(np.diff(idx) > 0) and not np.any(labels[idx] == 255)
    for c in (0, 1, 4, 6):
        m = labels[idx] == c
        assert abs(w[m].astype(np.float64).sum() - (labels == c).sum()) < 0.05   # weighted counts = true class totals
    assert (labels[idx] == 4).sum() == 5_000 and (labels[idx] == 1).sum() == 1_000


def test_perturbations_and_window_slice():
    S = torch.tensor([[10.0, 0, 50], [0.0, 0, 50], [30.0, 0, 50], [20.0, 0, 50]])   # rows: b, a, d, c
    R = torch.eye(3).expand(4, 3, 3).clone()
    ids = ["b", "a", "d", "c"]                                        # capture order a(0) b(10) c(20) d(30) along +x
    s, d = P.along_track(S, ids)
    assert torch.allclose(s[[1, 0, 3, 2]], torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]), atol=1e-3) and abs(d[0] - 1) < 1e-3
    S2, R2 = P.drift_cams(S, R, ids, max_t=0.3, max_r_deg=1.0, sign=1.0)
    assert torch.allclose(S2[1], S[1]) and abs((S2[2] - S[2])[0] - 0.3) < 1e-4       # first camera unmoved, last moved by max_t
    assert torch.allclose(R2 @ R2.transpose(1, 2), torch.eye(3).expand(4, 3, 3), atol=1e-5)
    S0, R0 = P.perturbed_cams(S, R, ids, "trans", 0.0, seed=3)
    assert torch.equal(S0, S) and torch.equal(R0, R)
    Sa, Ra = P.perturbed_cams(S, R, ids, "rot", 0.5, seed=3); Sb, Rb = P.perturbed_cams(S, R, ids, "rot", 0.5, seed=3)
    assert torch.equal(Ra, Rb) and torch.equal(Sa, S) and not torch.allclose(Ra, R)   # deterministic in the seed
    Sc, _ = P.perturbed_cams(S, R, ids, "trans", 0.1, seed=4)
    assert 0.02 < (Sc - S).norm(dim=1).mean() < 0.5
    # the drift has one degree of freedom, its direction: draw 0 = +, draw 1 = -, whatever the seed
    Sp, Rp = P.perturbed_cams(S, R, ids, "drift", 0.25, seed=1, draw=0); Sm, Rm = P.perturbed_cams(S, R, ids, "drift", 0.25, seed=9, draw=1)
    assert torch.allclose(Sp - S, -(Sm - S), atol=1e-6) and abs((Sp - S).abs().max() - 0.25) < 1e-4
    assert torch.allclose(Rp, Rm.transpose(1, 2), atol=1e-6) and not torch.allclose(Rp, R)
    assert P.draws_for("drift", 0.25, 10) == 2 and P.draws_for("trans", 0.25, 10) == 10 and P.draws_for("rot", 0.0, 10) == 1
    assert P.draws_for("drift", 0.25, 1) == 1
    f5 = torch.arange(2 * 3 * 25 * 4, dtype=torch.float32).reshape(2, 3, 25, 4)
    f3 = P.window_slice(f5, 5, 3); f1 = P.window_slice(f5, 5, 1)
    assert f3.shape == (2, 3, 9, 4) and f1.shape == (2, 3, 4)
    assert torch.equal(f1, f5[:, :, 12]) and torch.equal(f3[:, :, 4], f5[:, :, 12])
    assert P.window_slice(None, 3, 1) is None and P.window_slice(f3, 3, 3) is f3
    assert abs(P.robustness_score([0, 1, 2], [0.5, 0.5, 0.5]) - 1.0) < 1e-9
    assert abs(P.robustness_score([0, 1, 2], [0.5, 0.25, 0.0]) - 0.5) < 1e-9
    assert np.isnan(P.robustness_score([0, 1], [0.0, 0.0]))


def test_run_protocol_end_to_end(tmp_path):
    cache, cache2d, data = write_zone(str(tmp_path), n=6000)
    seg = DummySeg()
    backbone = FrozenBackbone("", "", device="cpu", model=seg)
    heads = {"late": LateFusionHead(), "concat": ConcatHead(), "win3": WindowAttnHead(k=3)}
    models = P.ProtocolModels(heads, device="cpu")
    assert models.names == ["lidar", "imagevote", "late", "concat", "win3"] and models.max_window == 3

    def loader(zone):
        ctx = load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), zone, k=3, need_depths=True, device="cpu", verbose=False)
        return ctx, None
    sweeps = {"trans": [0.0, 0.05, 0.5], "rot": [0.0, 0.5], "drift": [0.0, 0.25]}
    cfg = {"radius": 25.0, "grid": 0.5, "eval_max_points": 4000}
    out = str(tmp_path / "proto")
    res = P.run_protocol(models, loader, ["zz"], out, sweeps, draws=2, mode="fast", seed=0, k=3, backbone=backbone, cfg=cfg,
                         cap=800, cap_full=300, verbose=False)
    assert os.path.exists(tmp_path / "proto" / "zz" / "subset.npz") and os.path.exists(tmp_path / "proto" / "results.json")
    sub = P.load_subset(str(tmp_path / "proto" / "zz" / "subset.npz"))
    assert sub["feat"].shape == (len(sub["idx"]), 72) and sub["logits"].shape[1] == 11 and sub["img"].shape[1] == 3
    # 5 models x [trans: 1 + 2 + 2, rot: 1 + 2, drift: 1 + 2] evaluations of the single zone
    assert len(res["runs"]) == 5 * (5 + 3 + 3)
    lidar = [r for r in res["runs"] if r["model"] == "lidar"]
    assert all(np.allclose(np.asarray(r["cm"]), np.asarray(lidar[0]["cm"])) for r in lidar)   # reference is constant
    cm0 = np.asarray(lidar[0]["cm"]); labels = np.load(os.path.join(cache, "zz", "points.npz"))["labels"]
    for c in (0, 1, 4, 6, 8):                                   # weighted rows = full-zone class totals (cells stored to 0.1)
        assert abs(cm0[c].sum() - (labels == c).sum()) < 1.0
    assert res["meta"]["version"] == P.PROTOCOL_VERSION and res["meta"]["drift_draws"] == 2
    drift = sorted(r["draw"] for r in res["runs"] if r["model"] == "imagevote" and r["sweep"] == "drift" and r["level"] == 0.25)
    assert drift == [0, 1]                                      # one evaluation per direction
    # coverage must not increase with noise, and the pure-image model must change under a large perturbation
    cov = {(r["sweep"], r["level"]): r["coverage"] for r in res["runs"] if r["model"] == "imagevote"}
    assert cov[("trans", 0.5)] <= cov[("trans", 0.0)] + 1e-9
    # mode "fast" keeps the visibility of the clean association; "fast_retest" also tests the new pixel against the
    # clean depth maps and can only lose more pairs (same cameras: the draws are seeded by sweep name + level value)
    resr = P.run_protocol(models, loader, ["zz"], str(tmp_path / "proto_retest"), {"trans": [0.0, 0.5]}, draws=2, mode="fast_retest",
                          seed=0, k=3, subset_dir=out, verbose=False)
    covr = {(r["level"], r["draw"]): r["coverage"] for r in resr["runs"] if r["model"] == "imagevote"}
    covf = {(r["level"], r["draw"]): r["coverage"] for r in res["runs"] if r["model"] == "imagevote" and r["sweep"] == "trans"}
    assert covr[(0.0, 0)] == covf[(0.0, 0)] and all(covr[(0.5, d)] <= covf[(0.5, d)] + 1e-9 for d in (0, 1))
    assert any(covr[(0.5, d)] < covf[(0.5, d)] - 1e-6 for d in (0, 1))
    iv = {r["level"]: np.asarray(r["cm"]) for r in res["runs"] if r["model"] == "imagevote" and r["sweep"] == "trans" and r["draw"] == 0}
    assert not np.allclose(iv[0.0], iv[0.5])
    summ = json.load(open(tmp_path / "proto" / "summary.json"))
    assert os.path.exists(tmp_path / "proto" / "curves.csv") and os.path.exists(tmp_path / "proto" / "scores.csv")
    # per (zone, sweep, level, draw): how far the stored pairs moved in the image (0 at level 0, more at larger errors)
    dr = {(d["sweep"], d["level"], d["draw"]): d for d in res["draws"]}
    assert len(res["draws"]) == 5 + 3 + 3 and all(d["zone"] == "zz" for d in res["draws"])
    assert dr[("trans", 0.0, 0)]["shift_px"] == 0.0 and dr[("rot", 0.0, 0)]["shift_px_electrical"] == 0.0
    assert 0 < dr[("trans", 0.05, 0)]["shift_px"] < dr[("trans", 0.5, 0)]["shift_px"] <= dr[("trans", 0.5, 0)]["shift_px_p90"]
    assert dr[("rot", 0.5, 1)]["shift_px"] > 0 and dr[("drift", 0.25, 0)]["shift_px"] > 0
    assert abs(dr[("trans", 0.5, 0)]["coverage"] - cov[("trans", 0.5)]) < 0.2
    disp = {(d["sweep"], d["level"]): d for d in summ["displacement"]}
    assert len(disp) == 3 + 2 + 2 and disp[("trans", 0.5)]["n"] == 2 and disp[("trans", 0.0)]["shift_px"] == 0.0
    assert abs(disp[("trans", 0.5)]["shift_px"] - np.mean([dr[("trans", 0.5, d)]["shift_px"] for d in (0, 1)])) < 1e-9
    # the displacement grows with the error (ten times the sigma, different random cameras: a wide bracket)
    assert 3 * disp[("trans", 0.05)]["shift_px"] < disp[("trans", 0.5)]["shift_px"] < 30 * disp[("trans", 0.05)]["shift_px"]
    # the pooled confusion matrices live in their own compact file; results.json has a gzip copy with the same content
    assert "confusion" not in summ
    conf = json.load(open(tmp_path / "proto" / "confusion.json"))
    assert conf["class_names"] == P.CLASS_NAMES and len(conf["rows"]) == 5 * (3 + 2 + 2)
    c0 = [r for r in conf["rows"] if r["model"] == "lidar" and r["sweep"] == "trans" and r["level"] == 0.5][0]
    assert c0["draws"] == 2 and np.allclose(np.asarray(c0["cm"]), cm0, atol=0.06)
    import gzip
    with gzip.open(tmp_path / "proto" / "results.json.gz", "rt") as fh:
        assert json.load(fh) == json.load(open(tmp_path / "proto" / "results.json"))
    sc = {(s["model"], s["sweep"], s["class"]): s for s in summ["scores"]}
    assert abs(sc[("lidar", "trans", "mean_all")]["R"] - 1.0) < 1e-9 and abs(sc[("lidar", "rot", "mean_all")]["drop_ref"]) < 1e-9
    assert sc[("imagevote", "trans", "mean_all")]["ref_level"] == 0.05
    cur = [c for c in summ["curves"] if c["model"] == "win3" and c["sweep"] == "trans" and c["class"] == "mean_all"]
    assert [c["level"] for c in cur] == [0.0, 0.05, 0.5] and cur[1]["draws"] == 2 and cur[0]["draws"] == 1
    # resume: a second call skips the finished zone and leaves the results unchanged
    res2 = P.run_protocol(models, loader, ["zz"], out, sweeps, draws=2, mode="fast", seed=0, k=3, verbose=False)
    assert len(res2["runs"]) == len(res["runs"])
    # ... but not with other settings: zones of one results file must share models, sweeps, draws, mode and version
    with pytest.raises(ValueError, match="other settings"):
        P.run_protocol(models, loader, ["zz"], out, sweeps, draws=3, mode="fast", seed=0, k=3, verbose=False)
    with pytest.raises(ValueError, match="models"):
        P.run_protocol(P.ProtocolModels({"late": heads["late"]}, device="cpu"), loader, ["zz"], out, sweeps, draws=2, mode="fast",
                       seed=0, k=3, verbose=False)
    old = dict(res["meta"]); old.pop("version")                # a file of protocol version 1 (random drift directions)
    assert "version" in P.meta_conflict(old, res["meta"]) and P.meta_conflict(res["meta"], res["meta"]) is None
    # a results.json written before the displacement was recorded (update32) is still summarised and continued
    legacy = str(tmp_path / "proto_legacy"); os.makedirs(legacy)
    with open(os.path.join(legacy, "results.json"), "w") as fh:
        json.dump({"meta": res["meta"], "runs": res["runs"]}, fh)
    res3 = P.run_protocol(models, loader, ["zz"], legacy, sweeps, draws=2, mode="fast", seed=0, k=3, subset_dir=out, verbose=False)
    summ3 = json.load(open(os.path.join(legacy, "summary.json")))
    assert res3["draws"] == [] and summ3["displacement"] == [] and summ3["curves"] == summ["curves"]
    assert os.path.exists(os.path.join(legacy, "confusion.json"))
    q = P.weighted_quantiles(torch.tensor([3.0, 1.0, 2.0, 10.0]), torch.tensor([1.0, 1.0, 1.0, 97.0]), (0.01, 0.5, 0.9))
    assert q == [1.0, 10.0, 10.0] and np.isnan(P.weighted_quantiles(torch.zeros(0), torch.zeros(0), (0.5,))[0])
    # fewer views at test time: the nearest valid ones are kept, whatever slot they were stored in
    rr = {"img": torch.tensor([[5, 6, 7], [1, 2, 3]]), "u": torch.tensor([[50., 60., 70.], [10., 20., 30.]]), "v": torch.zeros(2, 3),
          "mask": torch.tensor([[True, True, True], [False, True, True]]),
          "desc": torch.tensor([[[3.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0]], [[0.5, 0, 0, 0, 0], [9.0, 0, 0, 0, 0], [4.0, 0, 0, 0, 0]]])}
    n2 = P.nearest_views(rr, 2); n1 = P.nearest_views(rr, 1)
    assert n2["img"].tolist() == [[6, 7], [3, 2]] and n2["u"].tolist() == [[60., 70.], [30., 20.]] and n2["mask"].all()
    assert n1["img"].tolist() == [[6], [3]] and n1["desc"].shape == (2, 1, 5) and P.nearest_views(rr, 3) is rr
    one = str(tmp_path / "proto_v1")                           # the same protocol with one view per point
    res1 = P.run_protocol(models, loader, ["zz"], one, {"trans": [0.0, 0.5]}, draws=2, mode="fast", seed=0, k=1, subset_dir=out, verbose=False)
    assert res1["meta"]["views"] == 1 and res["meta"]["views"] == 3
    c1 = {(r["model"], r["level"], r["draw"]): np.asarray(r["cm"]) for r in res1["runs"]}
    c3 = {(r["model"], r["level"], r["draw"]): np.asarray(r["cm"]) for r in res["runs"] if r["sweep"] == "trans"}
    assert np.allclose(c1[("lidar", 0.0, 0)], c3[("lidar", 0.0, 0)], atol=0.06)                  # LiDAR-only does not read the views
    assert not np.allclose(c1[("imagevote", 0.0, 0)], c3[("imagevote", 0.0, 0)], atol=0.06)      # one vote instead of up to three
    with pytest.raises(ValueError, match="views"):             # another number of views is another run
        P.run_protocol(models, loader, ["zz"], one, {"trans": [0.0, 0.5]}, draws=2, mode="fast", seed=0, k=2, subset_dir=out, verbose=False)
    assert P.meta_conflict({k: v for k, v in res["meta"].items() if k != "views"}, dict(res["meta"], views=8)) is None   # files without the key had 8
    # a subset without electrical points: the electrical statistic is null (valid JSON), the others are numbers
    ctxs, _ = loader("zz"); subs = P.load_subset(os.path.join(out, "zz", "subset.npz")); rws = P.subset_rows(subs, "cpu")
    xyzs = torch.from_numpy(ctxs.zone.xyz[subs["idx"]]); S2, R2 = P.perturbed_cams(ctxs.S, ctxs.R, ctxs.assoc.image_ids, "trans", 0.5, 1)
    sh = P.pixel_shift(rws, xyzs, ctxs.S, ctxs.R, S2, R2, ctxs.cal, torch.from_numpy(subs["w"]), torch.full((len(xyzs),), 6), stride=1)
    assert sh["shift_px_electrical"] is None and sh["shift_px"] > 0 and json.loads(json.dumps(sh, allow_nan=False)) == sh
    same = P.pixel_shift(rws, xyzs, ctxs.S, ctxs.R, ctxs.S, ctxs.R, ctxs.cal, torch.from_numpy(subs["w"]), torch.from_numpy(subs["labels"].astype(np.int64)))
    assert same["shift_px"] == 0.0 and same["shift_px_p90"] == 0.0                   # same cameras: nothing moves
    # a 5x5 window head is evaluated in smaller steps (same predictions as in one step)
    m5 = P.ProtocolModels({"win5": WindowAttnHead(k=5), "late": heads["late"]}, device="cpu")
    ctx5, _ = loader("zz"); sub5 = P.load_subset(os.path.join(out, "zz", "subset.npz")); rows5 = P.subset_rows(sub5, "cpu")
    lg5, ft5 = torch.from_numpy(sub5["logits"]), torch.from_numpy(sub5["feat"])
    pa = m5.predict(ctx5.maps, rows5, lg5, ft5, chunk=200_000); pb = m5.predict(ctx5.maps, rows5, lg5, ft5, chunk=30_000)
    assert m5.max_window == 5 and all(torch.equal(pa[n], pb[n]) for n in m5.names)
    # exact mode on a reduced grid runs and agrees with fast at level 0
    out2 = str(tmp_path / "proto_exact")
    rese = P.run_protocol(models, loader, ["zz"], out2, {"trans": [0.0, 0.5]}, draws=1, mode="exact", seed=0, k=3,
                          subset_dir=str(tmp_path / "proto"), verbose=False)
    e0 = [r for r in rese["runs"] if r["model"] == "concat" and r["level"] == 0.0][0]
    f0 = [r for r in res["runs"] if r["model"] == "concat" and r["sweep"] == "trans" and r["level"] == 0.0][0]
    assert np.allclose(np.asarray(e0["cm"]), np.asarray(f0["cm"]))
    # draws are seeded by sweep name and level value: the exact run (other level list) used the cameras of the fast run
    assert P.draw_seed(0, "trans", 0.5, 0) == P.draw_seed(0, "trans", 0.5, 0) != P.draw_seed(0, "trans", 0.5, 1)
    assert len({P.draw_seed(0, sw, lv, d) for sw in ("trans", "rot", "drift") for lv in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0) for d in range(10)}) == 180
    assert P.draw_seed(1, "trans", 0.5, 0) != P.draw_seed(0, "trans", 0.5, 0) != P.draw_seed(0, "rot", 0.5, 0)
    rows = P.compare_runs(res, rese)                                   # fast vs exact on the evaluations they share
    assert {(r["model"], r["sweep"], r["level"]) for r in rows} == {(m, "trans", lv) for m in models.names for lv in (0.0, 0.5)}
    for r in rows:
        assert r["n"] == 1 and r["zones"] == ["zz"] and r["draws"] == [0] and 0.0 <= r["coverage_b"] <= 1.0
        if r["level"] == 0.0 or r["model"] == "lidar":                 # same rows / no image evidence -> identical
            assert r["iou_a"] == r["iou_b"] and abs(r["mean_all_b"] - r["mean_all_a"]) < 1e-12
    big = [r for r in rows if r["level"] == 0.5 and r["model"] == "imagevote"][0]
    assert abs(big["coverage_b"] - big["coverage_a"]) < 0.25 and big["max_diff_class"] in P.CLASS_NAMES
    # the scripts on those two runs: tables (no crash on absent classes) and the comparison file
    import subprocess, sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    r1 = subprocess.run([sys.executable, os.path.join(root, "scripts", "compare_protocol.py"), "--a", out, "--b", out2],
                        capture_output=True, text=True)
    assert r1.returncode == 0 and "largest class difference" in r1.stdout and os.path.exists(os.path.join(out2, "compare.json")), r1.stderr[-800:]
    # run as a notebook runs it: the kernel exports its inline backend, which the subprocess cannot import
    r2 = subprocess.run([sys.executable, os.path.join(root, "scripts", "plot_protocol.py"), "--summary", os.path.join(out, "summary.json"),
                         "--out", str(tmp_path / "fig.png")], capture_output=True, text=True,
                        env=dict(os.environ, MPLBACKEND="module://no_such_inline_backend"))
    assert r2.returncode == 0 and "IoU lost at the reference level" in r2.stdout and os.path.exists(tmp_path / "fig.png"), r2.stderr[-800:]
    assert "median displacement of a projected point [px]" in r2.stdout
    r3 = subprocess.run([sys.executable, os.path.join(root, "scripts", "plot_protocol.py"), "--summary", os.path.join(legacy, "summary.json"),
                         "--out", str(tmp_path / "fig3.png"), "--tables", "compact"], capture_output=True, text=True)
    assert r3.returncode == 0 and "median displacement" not in r3.stdout and "all classes / electrical classes" in r3.stdout, r3.stderr[-800:]


def test_on_draw_hook_saves_the_predictions_behind_the_results(tmp_path):
    """run_protocol(on_draw=CropDump): the crops hold draw 0 of the wanted levels, their statistics are the confusion
    matrices of results.json, and the results are the same with and without the hook."""
    from gridnet import qualitative as Q
    cache, cache2d, data = write_zone(str(tmp_path), n=6000)
    backbone = FrozenBackbone("", "", device="cpu", model=DummySeg())
    torch.manual_seed(0)
    models = P.ProtocolModels({"late": LateFusionHead(), "concat": ConcatHead()}, device="cpu")

    def loader(zone):
        return load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), zone, k=3, device="cpu", verbose=False), None
    sweeps = {"trans": [0.0, 0.5], "rot": [0.0, 1.0]}
    cfg = {"radius": 25.0, "grid": 0.5, "eval_max_points": 4000}
    plain = P.run_protocol(models, loader, ["zz"], str(tmp_path / "plain"), sweeps, draws=2, mode="exact", seed=0, k=3, backbone=backbone,
                           cfg=cfg, cap=800, cap_full=300, verbose=False)
    calls = []
    dump = Q.CropDump(str(tmp_path / "crops"), [("trans", 0.0), ("trans", 0.5), ("rot", 1.0)], models=["lidar", "concat"], radius=1e4,
                      pylons=2, voxel=0.0, max_points=10**9)

    def hook(**kw):
        calls.append((kw["sweep"], kw["level"], kw["draw"], sorted(kw["preds"])))
        dump(**kw)
    res = P.run_protocol(models, loader, ["zz"], str(tmp_path / "hook"), sweeps, draws=2, mode="exact", seed=0, k=3,
                         subset_dir=str(tmp_path / "plain"), verbose=False, on_draw=hook)
    assert [r["cm"] for r in res["runs"]] == [r["cm"] for r in plain["runs"]]          # the hook changes nothing
    assert len(calls) == 3 + 3 and calls[0] == ("trans", 0.0, 0, ["concat", "imagevote", "late", "lidar"])   # every evaluation, every model
    paths = dump.paths("zz")
    assert len(paths) == 1                                      # no pylon with 500 points in the synthetic zone: one crop at the power-line points
    crop = Q.load_crop(paths[0])
    sub = P.load_subset(str(tmp_path / "plain" / "zz" / "subset.npz"))
    xyz = np.load(os.path.join(cache, "zz", "points.npz"))["xyz"][sub["idx"]].astype(np.float64)
    rel = np.column_stack([xyz[:, :2] - crop["centre"], xyz[:, 2] - crop["z0"]])
    order = Q.spatial_order(np.round(rel * 100.0).astype(np.int64))                  # a crop file is in spatial order
    assert crop["n_subset"] == len(sub["idx"]) == len(crop["labels"]) and np.array_equal(crop["labels"], sub["labels"][order])
    assert sorted(crop["preds"]) == ["concat|rot|1", "concat|trans|0", "concat|trans|0.5", "lidar|rot|1", "lidar|trans|0", "lidar|trans|0.5"]
    for k, cm in crop["cm"].items():                            # the crop covers the zone: its statistics are the results (stored to 0.1)
        m, s, lv = Q.split_key(k)
        r = [x for x in res["runs"] if (x["model"], x["sweep"], x["level"], x["draw"]) == (m, s, lv, 0)][0]
        assert np.allclose(cm, np.asarray(r["cm"]), atol=0.06), k
        assert np.allclose(Q.weighted_cm(crop["labels"], crop["preds"][k], sub["w"][order]), cm, rtol=1e-5), k   # and the saved points give them back
    assert np.array_equal(crop["preds"]["lidar|trans|0"], crop["preds"]["lidar|rot|1"])   # LiDAR-only does not see the cameras
    assert not np.array_equal(crop["preds"]["concat|trans|0"], crop["preds"]["concat|rot|1"])
    assert np.abs(crop["xyz"] - rel[order]).max() <= 0.0051
    # a finished zone is not evaluated again: the hook is not called and the files stay
    stamp = os.path.getmtime(paths[0]); calls.clear()
    P.run_protocol(models, loader, ["zz"], str(tmp_path / "hook"), sweeps, draws=2, mode="exact", seed=0, k=3,
                   subset_dir=str(tmp_path / "plain"), verbose=False, on_draw=hook)
    assert calls == [] and os.path.getmtime(paths[0]) == stamp


def test_shared_error_sweeps(tmp_path):
    """trans_shared / rot_shared: ONE error for all cameras of a zone, drawn from the distribution of the i.i.d. sweep;
    the seeds of the sweeps that existing runs used do not move."""
    g = torch.Generator().manual_seed(1)
    S = torch.randn(9, 3, generator=g) * 20; R = torch.linalg.qr(torch.randn(9, 3, 3, generator=g))[0]
    ids = [f"img{i}" for i in range(9)]
    St, Rt = P.perturbed_cams(S, R, ids, "trans_shared", 0.5, seed=5)
    d = St - S
    assert torch.allclose(d, d[:1].expand_as(d), atol=1e-5) and torch.equal(Rt, R) and 0.05 < d[0].norm() < 3.0
    S1, _ = P.perturbed_cams(S[:1], R[:1], ids[:1], "trans", 0.5, seed=5)          # = the i.i.d. draw of ONE camera
    assert torch.allclose(S1 - S[:1], d[:1], atol=1e-5)
    Sr, Rr = P.perturbed_cams(S, R, ids, "rot_shared", 1.0, seed=5)
    dR = Rr @ R.transpose(1, 2)                                                    # the rotation applied in the camera frame
    assert torch.equal(Sr, S) and torch.allclose(dR, dR[:1].expand_as(dR), atol=1e-5)
    assert torch.allclose(Rr @ Rr.transpose(1, 2), torch.eye(3).expand(9, 3, 3), atol=1e-5)
    _, R1 = P.perturbed_cams(S[:1], R[:1], ids[:1], "rot", 1.0, seed=5)
    assert torch.allclose(R1, Rr[:1], atol=1e-5)
    S0, R0 = P.perturbed_cams(S, R, ids, "rot_shared", 0.0, seed=5)
    assert torch.equal(S0, S) and torch.equal(R0, R)
    # the marginal distribution is that of the i.i.d. sweep: per-axis standard deviation = the level
    draws = torch.stack([P.perturbed_cams(S[:1], R[:1], ids[:1], "trans_shared", 0.25, seed=s)[0][0] - S[0] for s in range(400)])
    assert abs(float(draws.std(0).mean()) - 0.25) < 0.03 and abs(float(draws.mean())) < 0.04
    assert P.draws_for("trans_shared", 0.25, 10) == 10 and P.draws_for("rot_shared", 0.0, 10) == 1
    assert set(P.SHARED_SWEEPS) <= set(P.SWEEPS) and P.SWEEPS["trans_shared"] == P.SWEEPS["trans"] and P.SWEEPS["rot_shared"] == P.SWEEPS["rot"]
    # one error per zone: the seed of a shared sweep depends on the zone, that of every other sweep does not ...
    assert P.draw_seed(0, "trans_shared", 0.5, 0, zone="t1z8") != P.draw_seed(0, "trans_shared", 0.5, 0, zone="t5b5")
    assert P.draw_seed(0, "rot_shared", 0.5, 0, zone="t1z8") == P.draw_seed(0, "rot_shared", 0.5, 0, zone="t1z8")
    assert P.draw_seed(0, "trans", 0.5, 0, zone="t1z8") == P.draw_seed(0, "trans", 0.5, 0)
    # ... and the seeds of the runs made so far (exact_v1, qual_v1, the heads labs) are what they were (values of update40)
    assert P.draw_seed(0, "trans", 0.5, 0) == 1549054884 and P.draw_seed(0, "rot", 1.0, 3) == 1721107792
    assert P.draw_seed(0, "drift", 0.25, 1) == 3209485136 and P.draw_seed(7, "rot", 0.05, 2) == 561529245

    # end to end on a small zone: the shared sweeps run through the same loop, the clean level equals that of the i.i.d. run
    cache, cache2d, data = write_zone(str(tmp_path), n=6000)
    backbone = FrozenBackbone("", "", device="cpu", model=DummySeg())
    models = P.ProtocolModels({"late": LateFusionHead(), "concat": ConcatHead()}, device="cpu")

    def loader(zone):
        ctx = load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), zone, k=3, need_depths=True, device="cpu", verbose=False)
        return ctx, None
    cfg = {"radius": 25.0, "grid": 0.5, "eval_max_points": 4000}
    iid = P.run_protocol(models, loader, ["zz"], str(tmp_path / "iid"), {"trans": [0.0, 0.5], "rot": [0.0, 1.0]}, draws=3, mode="exact",
                         seed=0, k=3, backbone=backbone, cfg=cfg, cap=800, cap_full=300, verbose=False)
    sh = P.run_protocol(models, loader, ["zz"], str(tmp_path / "shared"), {"trans_shared": [0.0, 0.5], "rot_shared": [0.0, 1.0]}, draws=3,
                        mode="exact", seed=0, k=3, subset_dir=str(tmp_path / "iid"), verbose=False)
    assert len(sh["runs"]) == 4 * (1 + 3 + 1 + 3) and sh["meta"]["reference"]["rot_shared"] == 0.10
    for m in models.names:
        a = [r for r in sh["runs"] if r["model"] == m and r["level"] == 0.0]
        b = [r for r in iid["runs"] if r["model"] == m and r["sweep"] == "trans" and r["level"] == 0.0][0]
        assert all(np.allclose(np.asarray(x["cm"]), np.asarray(b["cm"])) for x in a)            # same points, same heads
    moved = [d for d in sh["draws"] if d["level"] > 0]
    assert len(moved) == 6 and all(d["shift_px"] is None or d["shift_px"] > 0 for d in moved)
    vote = {(r["sweep"], r["draw"]): np.asarray(r["cm"]) for r in sh["runs"] if r["model"] == "imagevote" and r["level"] > 0}
    assert not np.allclose(vote[("trans_shared", 0)], vote[("trans_shared", 1)])               # the draws are different errors
    summ = P.summarize(sh)
    assert {s["sweep"] for s in summ["scores"]} == {"trans_shared", "rot_shared"}
    assert {(d["sweep"], d["level"]) for d in summ["displacement"]} == {("trans_shared", 0.0), ("trans_shared", 0.5), ("rot_shared", 0.0), ("rot_shared", 1.0)}
    # the comparison script on the two runs: check line, both tables, the file
    import subprocess, sys
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    r = subprocess.run([sys.executable, os.path.join(here, "scripts", "compare_shared.py"), "--shared", str(tmp_path / "shared"),
                        "--iid", str(tmp_path / "iid"), "--out", str(tmp_path / "cmp.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "largest difference between the runs over the models they share: 0.000" in r.stdout and "as it must be" in r.stdout
    assert "LiDAR-only" in r.stdout and "ImageVote" in r.stdout and "50 cm" in r.stdout and "1 deg" in r.stdout
    doc = json.load(open(tmp_path / "cmp.json"))
    assert doc["clean_check"] < 1e-9 and {x["head"] for x in doc["rows"]} == {"LiDAR-only", "ImageVote"}   # no head of the paper here
    lid = [x for x in doc["rows"] if x["head"] == "LiDAR-only" and x["class"] == "mean_all"]
    assert all(abs(x["iid"] - x["shared"]) < 1e-9 for x in lid) and {x["level"] for x in lid} == {0.0, 0.5, 1.0}
