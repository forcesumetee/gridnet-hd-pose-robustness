import json
import os
import sys

import numpy as np
import pytest
torch = pytest.importorskip("torch")   # GPU-side tests: skipped on the torch-less Mac venv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fusion import DummySeg, write_zone  # noqa: E402

from gridnet.train import bank as B  # noqa: E402
from gridnet.train import protocol as P  # noqa: E402
from gridnet.train.engine_fusion import FrozenBackbone, class_weights_from_counts, label_counts, load_zone_context  # noqa: E402
from gridnet.train.fusion import ConcatHead, WindowAttnHead, assoc_rows, build_head, drop_lidar  # noqa: E402


def load_script(name):
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_parse_specs_and_head_variants():
    specs = B.parse_specs("late_b:late concat_ld:concat:lidar_drop=0.5,seed=2; win3p:window:k=5,lidar_input=prob,pose_t=0.25,pose_r=0.1")
    assert [s.name for s in specs] == ["late_b", "concat_ld", "win3p"]
    assert specs[1].lidar_drop == 0.5 and specs[1].seed == 2 and specs[1].head == "concat"
    assert specs[2].k == 5 and specs[2].lidar_input == "prob" and specs[2].pose_t == 0.25 and specs[2].group() == (0, 0.25, 0.1, 0.5, 1.0)
    bal = B.parse_specs("late_h:late:natural_frac=0.5")[0]            # default 1.0 = batches at the natural class frequencies
    assert bal.natural_frac == 0.5 and bal.group() != B.parse_specs("late_b:late")[0].group()        # own batches
    ip = B.parse_specs("c_ip:concat:image_input=prob")[0]
    assert ip.image_input == "prob" and ip.build().uses_feat is False and B.parse_specs("c:concat")[0].build().uses_feat is True
    for bad in ("x", "a:mlp", "a:late:k", "a:late:foo=1", "a:late b:late a:concat", "a:concat:lidar_input=logits", "a:late:seed=1:k=3",
                "a:window:image_input=prob", "a:concat:image_input=none"):
        with pytest.raises(ValueError):
            B.parse_specs(bad)
    # "prob" heads see the LiDAR branch only through its class probabilities
    torch.manual_seed(0)
    N, V = 50, 4
    logits, feat = torch.randn(N, 11), torch.randn(N, 72)
    img_prob = torch.rand(N, V, 11); img_feat = torch.randn(N, V, 32); win = torch.randn(N, V, 9, 32)
    desc = torch.rand(N, V, 5); mask = torch.rand(N, V) > 0.3
    hp = ConcatHead(lidar_input="prob").eval(); wp = WindowAttnHead(k=3, lidar_input="prob").eval()
    assert hp.net[0].in_features == 11 + 32 + 11 + 1 and wp.q.in_features == 11
    with torch.no_grad():
        assert torch.equal(hp(logits, feat, img_prob, img_feat, desc, mask), hp(logits, torch.zeros_like(feat), img_prob, img_feat, desc, mask))
        assert torch.equal(wp(logits, feat, img_prob, win, desc, mask), wp(logits, feat * 3, img_prob, win, desc, mask))
        hf = ConcatHead().eval()
        assert not torch.equal(hf(logits, feat, img_prob, img_feat, desc, mask), hf(logits, torch.zeros_like(feat), img_prob, img_feat, desc, mask))
    assert isinstance(build_head("late", lidar_input="prob", image_input="prob"), torch.nn.Module)     # options ignored by the late head
    # image_input="prob": LiDAR feature + image probabilities, no image feature (and no 2D feature maps needed)
    hi = ConcatHead(image_input="prob").eval()
    assert hi.net[0].in_features == 72 + 11 + 1 and not hi.uses_feat and "norm_i.weight" not in hi.state_dict()
    with torch.no_grad():
        assert torch.equal(hi(logits, feat, img_prob, None, desc, mask), hi(logits, feat, img_prob, img_feat, desc, mask))
        assert not torch.equal(hi(logits, feat, img_prob, None, desc, mask), hi(logits, feat * 2, img_prob, None, desc, mask))
    assert set(ConcatHead().state_dict()) == {"weighter.net.0.weight", "weighter.net.0.bias", "weighter.net.2.weight", "weighter.net.2.bias",
                                              "norm_l.weight", "norm_l.bias", "norm_i.weight", "norm_i.bias", "net.0.weight", "net.0.bias",
                                              "net.1.weight", "net.1.bias", "net.4.weight", "net.4.bias", "net.5.weight", "net.5.bias",
                                              "net.8.weight", "net.8.bias"}                # = the keys of concat_v1/best.pth
    with pytest.raises(ValueError):
        build_head("window", image_input="prob")
    # LiDAR dropout: dropped points get zero logits and features, the others are untouched
    g = torch.Generator().manual_seed(1)
    lg, ft, m = drop_lidar(logits, feat, 0.5, g)
    assert 5 < int(m.sum()) < 45 and torch.all(lg[m] == 0) and torch.all(ft[m] == 0)
    assert torch.equal(lg[~m], logits[~m]) and torch.equal(ft[~m], feat[~m])
    assert drop_lidar(logits, feat, 0.0)[2] is None and bool(drop_lidar(logits, feat, 1.0)[2].all())


def test_bank_sampling_mixes_natural_and_uniform():
    n = 20_000
    labels = np.zeros(n, np.uint8); labels[:1000] = 1                 # class 1: 1000 points kept fully (w = 1)
    w = np.full(n, 50.0, np.float32); w[:1000] = 1.0                 # class 0: every point stands for 50
    sub = {"zone": "zz", "idx": np.arange(n), "labels": labels, "w": w, "feat": np.zeros((n, 72), np.float16),
           "logits": np.zeros((n, 11), np.float16), "img": np.full((n, 3), -1, np.int16), "u": np.zeros((n, 3), np.uint16),
           "v": np.zeros((n, 3), np.uint16), "desc": np.zeros((n, 3, 5), np.float16), "image_ids": np.array(["a"])}
    bank = B.ZoneBank(sub, device="cpu")
    g = torch.Generator().manual_seed(0)
    share = lambda frac: float((bank.labels[bank.sample(200_000, g, frac)] == 1).float().mean())
    assert abs(share(0.0) - 1000 / n) < 0.004                         # uniform over the bank: 5 %
    assert abs(share(1.0) - 1000 / (1000 + 19_000 * 50)) < 0.001      # natural frequency: 0.1 %
    assert abs(share(0.5) - 0.5 * (0.05 + 1000 / 951_000)) < 0.004
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    assert torch.equal(bank.sample(100, g1), bank.sample(100, g2))


def test_bank_trainer_end_to_end(tmp_path):
    cache, cache2d, data = write_zone(str(tmp_path), n=6000)
    backbone = FrozenBackbone("", "", device="cpu", model=DummySeg())
    banks = str(tmp_path / "banks"); out_root = str(tmp_path / "fusion")
    ctx = load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), "zz", k=3, with_feat=False, device="cpu", verbose=False)
    sub = P.ensure_subset(ctx, os.path.join(banks, "zz", "subset.npz"), backbone, {"radius": 25.0, "grid": 0.5, "eval_max_points": 4000},
                          cap=800, cap_full=300, verbose=False)
    assert os.path.exists(os.path.join(banks, "zz", "subset.npz"))
    assert P.ensure_subset(ctx, os.path.join(banks, "zz", "subset.npz"))["idx"].shape == sub["idx"].shape     # reused, no backbone
    ra, rb = assoc_rows(ctx.assoc, sub["idx"], "cpu"), P.subset_rows(sub, "cpu")                              # same rows as the crop trainer
    assert set(ra) == set(rb) and all(torch.equal(ra[k], rb[k]) for k in ra)
    # class totals from the bank = totals from the point cloud
    assert np.allclose(B.bank_class_counts(banks, ["zz"]), label_counts(cache, ["zz"]), atol=0.5)
    table = B.format_counts(B.zone_class_counts(banks, ["zz"]))
    assert table.splitlines()[2].startswith("zz") and " . " in table + " "                                     # absent classes shown as '.'


    specs = B.parse_specs("late_b:late concat_b:concat concatp_ld:concat:lidar_input=prob,lidar_drop=0.5 "
                          "win3_aug:window:k=3,pose_t=0.3,pose_r=0.3,aug_prob=1.0 late_s1:late:seed=1 concat_ip:concat:image_input=prob")
    cfg = {"train_zones": ["zz"], "val_zones": ["zz"], "passes": 2, "its_per_zone": 3, "batch_points": 600, "chunk": 400, "seed_order": 0}
    class_w = class_weights_from_counts(B.bank_class_counts(banks, ["zz"]))
    loads = []

    def loader(zone, need_xyz, with_feat):
        loads.append((zone, need_xyz))
        return B.load_bank_context(cache, cache2d, data, banks, zone, need_xyz=need_xyz, with_feat=with_feat, device="cpu", verbose=False)
    tr = B.BankTrainer(specs, class_w, cfg, out_root, device="cpu")
    assert tr.need_xyz and tr.with_feat
    before = {r.spec.name: [p.clone() for p in r.head.parameters()] for r in tr.runs}
    assert tr.fit(loader) == "done"
    assert any(nd for _, nd in loads) and not all(nd for _, nd in loads)             # coordinates for training, not for validation
    for r in tr.runs:
        assert r.it == 6 and r.passes_done == 2 and r.best > 0
        assert any(not torch.equal(a, b) for a, b in zip(before[r.spec.name], r.head.parameters()))     # trained
        st = torch.load(os.path.join(out_root, r.spec.name, "best.pth"), map_location="cpu", weights_only=False)
        assert {"head", "cfg", "it", "best"} <= set(st) and st["cfg"]["trainer"] == "bank" and st["cfg"]["head"] == r.spec.head
        assert json.load(open(os.path.join(out_root, r.spec.name, "config.json")))["name"] == r.spec.name
        e = json.load(open(os.path.join(out_root, r.spec.name, "eval_bank", "eval.json")))
        assert e["tag"] == "best" and set(e["per_zone"]) == {"zz"} and len(e["confusion"]) == 11 and "imagevote" in e and "lidar_only" in e
        assert abs(e["mIoU"] - r.best) < 1e-9
    # different seeds -> different batches and initialisation -> different heads
    a, b = tr.runs[0], tr.runs[4]
    assert a.spec.name == "late_b" and b.spec.name == "late_s1"
    assert any(not torch.equal(p, q) for p, q in zip(a.head.parameters(), b.head.parameters()))

    # the bank evaluation of a head equals the protocol's level-0 evaluation of its checkpoint
    tr.load_best()
    res = tr.evaluate(loader, ["zz"], verbose=False)
    heads = P.ProtocolModels.from_checkpoints({r.spec.name: os.path.join(out_root, r.spec.name, "best.pth") for r in tr.runs}, device="cpu")
    assert heads["concatp_ld"].lidar_input == "prob" and heads["win3_aug"].window == 3
    assert heads["concat_ip"].image_input == "prob" and not heads["concat_ip"].uses_feat
    models = P.ProtocolModels(heads, device="cpu")
    c = loader("zz", False, True)
    preds = models.predict(c.maps, c.bank.rows, c.bank.logits, c.bank.feat)
    for name in [r.spec.name for r in tr.runs] + ["lidar", "imagevote"]:
        cm = P.weighted_confusion(c.bank.labels, preds[name], c.bank.w)
        assert np.allclose(cm, res[name]["cm"]), name

    # an existing checkpoint evaluated next to the trained heads (here: late_b's own best weights under another name)
    with pytest.raises(ValueError):
        tr.add_references({"late_b": build_head("late")})
    tr.add_references(P.ProtocolModels.from_checkpoints({"old_late": os.path.join(out_root, "late_b", "best.pth")}, device="cpu"))

    # diagnostics: three input cases on val (+ train zones), written per head
    diag = tr.diagnostics(loader, ["zz"])
    assert set(diag) == {"val", "val_no_lidar", "val_no_image", "train"}
    assert abs(diag["val"]["late_b"]["mIoU"] - diag["train"]["late_b"]["mIoU"]) < 1e-9          # same zone here
    assert diag["val_no_lidar"]["lidar"]["mIoU"] == diag["val"]["lidar"]["mIoU"]                # baselines are never ablated
    assert set(diag["val"]["late_b"]["per_zone"]) == {"zz"}
    for case in diag:                                                                           # same weights -> same numbers
        assert diag[case]["old_late"]["mIoU"] == diag[case]["late_b"]["mIoU"]
    assert not os.path.exists(os.path.join(out_root, "old_late"))                               # nothing written for a reference
    d = json.load(open(os.path.join(out_root, "concat_b", "eval_bank", "diagnostics.json")))
    assert set(d["val"]) == {"concat_b", "lidar", "imagevote"}
    text = B.format_table(diag["val"], ["lidar", "imagevote", "late_b", "concat_b"])
    assert text.splitlines()[2].startswith("lidar") and "late_b" in text and "pylon" in text

    # resume: everything is finished -> nothing is trained again; a new head joins and trains alone
    n = len(loads)
    tr2 = B.BankTrainer(specs, class_w, cfg, out_root, device="cpu")
    assert tr2.fit(loader) == "done" and len(loads) == n and all(r.it == 6 for r in tr2.runs)
    specs3 = specs + B.parse_specs("win5:window:k=5")
    tr3 = B.BankTrainer(specs3, class_w, cfg, out_root, device="cpu")
    assert tr3.fit(loader) == "done"
    assert [r.spec.name for r in tr3.runs if r.it == 6] == [s.name for s in specs3] and len(tr3.active) == 0
    assert os.path.exists(os.path.join(out_root, "win5", "best.pth"))
    # memory-only settings may change between sessions; a name is never reused for another experiment
    assert B.BankTrainer(specs, class_w, dict(cfg, chunk=300), out_root, device="cpu").fit(loader) == "done"
    for other_specs, other_cfg in ((B.parse_specs("late_b:late:lidar_drop=0.3"), cfg), (specs, dict(cfg, its_per_zone=4)),
                                   (B.parse_specs("late_b:concat"), cfg)):
        with pytest.raises(ValueError, match="another configuration"):
            B.BankTrainer(other_specs, class_w, other_cfg, out_root, device="cpu").fit(loader)
        with pytest.raises(ValueError, match="another configuration"):                           # the same check before any slow step
            B.check_names(other_specs, other_cfg, out_root)
    B.check_names(specs3, cfg, out_root)                                                         # unchanged requests pass
    B.check_names(B.parse_specs("brand_new:late"), cfg, out_root)
    assert not os.path.exists(os.path.join(out_root, "brand_new"))                               # nothing is created by looking
    before_w = torch.load(os.path.join(out_root, "late_b", "best.pth"), map_location="cpu", weights_only=False)["head"]
    after_w = tr.runs[0].best_state
    assert all(torch.equal(before_w[k], after_w[k]) for k in before_w)                           # untouched by the refused runs


def test_full_zone_evaluation_of_a_bank_head_and_training_guard(tmp_path, monkeypatch, capsys):
    """scripts/train_fusion.py --eval-only reads a bank-trained head (design from its checkpoint) and refuses to train on it."""
    cache, cache2d, data = write_zone(str(tmp_path), n=6000)
    backbone = FrozenBackbone("", "", device="cpu", model=DummySeg())
    banks, out_root, assoc = str(tmp_path / "banks"), str(tmp_path / "fusion"), str(tmp_path / "assoc")
    ctx = load_zone_context(cache, cache2d, data, assoc, "zz", k=3, with_feat=False, device="cpu", verbose=False, with_maps=False)
    P.ensure_subset(ctx, os.path.join(banks, "zz", "subset.npz"), backbone, {"radius": 25.0, "grid": 0.5, "eval_max_points": 4000},
                    cap=800, cap_full=300, verbose=False)
    specs = B.parse_specs("winp:window:k=3,lidar_input=prob")
    cfg = {"train_zones": ["zz"], "val_zones": ["zz"], "passes": 1, "its_per_zone": 3, "batch_points": 600, "chunk": 400, "seed_order": 0}
    tr = B.BankTrainer(specs, class_weights_from_counts(B.bank_class_counts(banks, ["zz"])), cfg, out_root, device="cpu")
    assert tr.fit(lambda z, nd, wf: B.load_bank_context(cache, cache2d, data, banks, z, need_xyz=nd, with_feat=wf, device="cpu",
                                                         verbose=False)) == "done"
    mod = load_script("train_fusion")
    monkeypatch.setattr(mod, "FrozenBackbone", lambda *a, **k: backbone)
    work = os.path.join(out_root, "winp")
    argv = ["train_fusion.py", "--cache", cache, "--cache2d", cache2d, "--data", data, "--assoc-dir", assoc, "--litept", "x",
            "--backbone-ckpt", "y", "--work-dir", work, "--val-zones", "zz", "--train-zones", "zz", "--grid", "0.5",
            "--eval-max-points", "4000", "--views", "3", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv + ["--eval-only"])          # --head is left at its default (concat): the checkpoint decides
    mod.main()
    e = json.load(open(os.path.join(work, "eval", "eval.json")))
    assert e["model"] == "window" and e["tag"] == "best" and e["it"] == 3 and set(e["per_zone"]) == {"zz"}
    assert os.path.exists(os.path.join(work, "eval", "zz", "fused_pred.npy"))
    assert "head window K=3" in capsys.readouterr().out
    bank_best = json.load(open(os.path.join(work, "eval_bank", "eval.json")))
    assert bank_best["model"] == "window" and bank_best["spec"]["lidar_input"] == "prob"       # the bank evaluation stays next to it
    monkeypatch.setattr(sys, "argv", argv)                              # training into the folder of a bank head: refused
    with pytest.raises(SystemExit, match="train_heads_bank"):
        mod.main()


def test_ensure_banks_builds_archives_and_reuses(tmp_path, monkeypatch):
    import argparse
    mod = load_script("train_heads_bank")
    cache, cache2d, data = write_zone(str(tmp_path), n=5000)
    built = []

    def fake_backbone(litept, ckpt, device="cpu", **kw):
        built.append(ckpt)
        return FrozenBackbone("", "", device="cpu", model=DummySeg())
    monkeypatch.setattr(mod, "FrozenBackbone", fake_backbone)
    a = argparse.Namespace(cache=cache, cache2d=cache2d, data=data, assoc_dir=str(tmp_path / "assoc"), subset_dir=str(tmp_path / "banks"),
                           subset_archive=str(tmp_path / "archive"), litept="", backbone_ckpt="ckpt.pth", device="cpu", radius=25.0,
                           grid=0.5, eval_max_points=4000, views=3, tau=0.5, assoc_chunk=20_000_000, subset_seed=0, cap=600, cap_full=300)
    mod.ensure_banks(a, ["zz"])                                         # first session: built with the backbone and archived
    local, arch = tmp_path / "banks" / "zz" / "subset.npz", tmp_path / "archive" / "zz" / "subset.npz"
    assert local.exists() and arch.exists() and built == ["ckpt.pth"]
    first = P.load_subset(str(local))
    assert len(first["idx"]) > 1000 and first["feat"].shape == (len(first["idx"]), 72)
    os.remove(local); mod.ensure_banks(a, ["zz"])                       # later session: copied from the archive, no backbone
    assert local.exists() and built == ["ckpt.pth"]
    again = P.load_subset(str(local))
    assert all(np.array_equal(first[k], again[k]) for k in first)
    mod.ensure_banks(a, ["zz"]); assert built == ["ckpt.pth"]           # present locally: nothing to do
    a.backbone_ckpt = None; os.remove(local); os.remove(arch)
    with pytest.raises(AssertionError):                                 # nothing to copy and nothing to build with
        mod.ensure_banks(a, ["zz"])
    # a bank is built without the zone's 2D maps on the device
    ctx = load_zone_context(cache, cache2d, data, str(tmp_path / "assoc"), "zz", k=3, with_feat=False, device="cpu", verbose=False,
                            with_maps=False)
    assert ctx.maps is None and ctx.assoc is not None and ctx.zone.n == 5000
