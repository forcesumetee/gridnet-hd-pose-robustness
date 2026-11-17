"""Training / evaluation of fusion heads on frozen backbones (zone-major, resume-safe).

Per zone the trainer holds: the LiDAR cache (`Zone`), the 2D maps on the GPU (`ZoneMaps`), the clean
association table (loaded from <assoc_dir>/<zone>/assoc.npz or computed on the GPU and saved), the camera
set and, when pose-noise augmentation is on, the clean depth maps. Crops come from the zone's own
`CropSampler`; the frozen LitePT-S gives per-point features/logits in an fp32 inference forward (spconv's
fp16 kernels misbehave in eval mode); the head is trained on a subset of the crop's labelled points, in
chunks with immediate backward so the K×K windows never need to be resident all at once.
"""
import contextlib
import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..camera import load_zone_cameras
from ..preprocess.association import Association
from .assoc_gpu import associate, depth_maps, zone_cams
from .data import ConfusionMeter, CropSampler, Zone, augment, cover_remaining, eval_windows, load_zone, make_batch
from .engine import CLASS_NAMES, build_model, to_device
from .fusion import ZoneMaps, assoc_rows, drop_lidar, lift, perturb_cams, reproject_rows
from .losses import SegLoss
from .maps2d import cached_image_ids


EVAL_SEED = 20261113      # generator state used for every validation / protocol forward pass


@contextlib.contextmanager
def fixed_torch_seed(seed: int = EVAL_SEED):
    """Run a block with torch's global generators seeded and restore them afterwards.
    LitePT (like PTv3) draws `torch.randperm` over its serialization orders in EVERY forward pass, also in eval
    mode, so the backbone's predictions are stochastic: three evaluations of the same checkpoints gave LiDAR-only
    69.88 / 70.02 / 69.97 mIoU (late_v1 run, 12 Nov 2026). Seeding makes validation numbers repeatable without
    touching the training random stream."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


# ----------------------------------------------------------------------------- frozen LiDAR backbone
class FrozenBackbone:
    """LitePT-S + its linear head from a LiDAR-only checkpoint, frozen; returns (feat (N,72), logits (N,11))."""

    def __init__(self, litept_root: str, ckpt: str, in_channels: int = 7, device: str = "cuda", model: Optional[nn.Module] = None):
        if model is None:
            model = build_model(litept_root, in_channels)
            st = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"])
            print(f"[fusion] frozen backbone from {ckpt} (it {st.get('it')}, val {st.get('best', float('nan')):.2f})", flush=True)
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.model.backbone(batch).feat.float()
        return feat, self.model.head(feat).float()


# ----------------------------------------------------------------------------- per-zone context
@dataclass
class ZoneContext:
    zone: Zone
    origin: np.ndarray
    maps: ZoneMaps
    assoc: Association
    cal: object
    S: torch.Tensor            # (n_img, 3) camera centres, zone-relative
    R: torch.Tensor            # (n_img, 3, 3)
    depths: Optional[torch.Tensor] = None   # (n_img, W*H) clean z-buffer, only when needed
    W: int = 0
    H: int = 0

    def free(self):
        self.maps = None; self.depths = None; self.S = None; self.R = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_zone_context(cache_dir: str, cache2d_dir: str, data_dir: str, assoc_dir: str, zone: str, k: int = 8,
                      tau: float = 0.5, chunk: int = 20_000_000, need_depths: bool = False, with_feat: bool = True,
                      device: str = "cuda", feature_set: str = "geo", verbose: bool = True,
                      assoc_source: str = "auto", with_maps: bool = True) -> ZoneContext:
    """Everything needed to train / evaluate heads on one zone. The association table comes from
    `assoc_source`: "archive" = <assoc_dir>/<zone>/assoc.npz when it exists, "compute" = rebuilt on the device,
    "auto" = compute on a GPU and archive on the CPU. On Colab the archive is a 4-10 GB file on the Drive mount
    that takes ~1 min to read, while rebuilding it on an A100 takes ~0.4 min (measured in the late_v1 run), so the
    archive is only written (once, when missing), not read, in GPU sessions. `with_maps=False` skips the 2D maps
    (ctx.maps is None): enough to compute backbone features of the zone, e.g. when a feature bank is built."""
    t0 = time.time()
    z = load_zone(cache_dir, zone, feature_set)
    with np.load(os.path.join(cache_dir, zone, "points.npz")) as p:
        origin = p["origin"]
        normals = p["normals"] if "normals" in p else None
    cal, poses = load_zone_cameras(os.path.join(data_dir, zone))
    have = set(cached_image_ids(cache2d_dir, zone))
    image_ids = sorted(i for i in poses if i in have)
    assert image_ids, f"no 2D cache for {zone} in {cache2d_dir}"
    apath = os.path.join(assoc_dir, zone, "assoc.npz")
    S, R = zone_cams(poses, image_ids, origin, device)
    depths = W = H = None
    assert assoc_source in ("auto", "archive", "compute"), assoc_source
    src = assoc_source
    if src == "auto":
        src = "compute" if str(device).startswith("cuda") else "archive"
    assoc = None
    if src == "archive" and os.path.exists(apath):
        assoc = Association.load(apath)
        if assoc.image_ids != image_ids:
            print(f"[fusion] WARNING: {apath} was built for a different image set ({len(assoc.image_ids)} vs "
                  f"{len(image_ids)} images) -> recomputing", flush=True)
            assoc = None
    if assoc is None or need_depths:
        depths, W, H = depth_maps(z.xyz, S, R, cal, chunk=chunk, verbose=verbose)
    if assoc is None:
        assoc = associate(z.xyz, origin, poses, cal, image_ids, k=k, tau=tau, normals=normals, chunk=chunk,
                          device=device, verbose=verbose, depths=depths)
        if not os.path.exists(apath):                       # archive: written once, never overwritten
            os.makedirs(os.path.dirname(apath), exist_ok=True)
            assoc.save(apath)
            if verbose:
                print(f"[fusion] association of {zone} saved to {apath}", flush=True)
    if not need_depths:
        depths = None
    maps = ZoneMaps(cache2d_dir, zone, image_ids, device=device, verbose=verbose, with_feat=with_feat) if with_maps else None
    if verbose:
        print(f"[fusion] {zone}: {z.n:,} points, {len(image_ids)} images, views/point {assoc.n_views.mean():.1f}, "
              f"no view {100 * (assoc.n_views == 0).mean():.1f} %  (loaded in {(time.time() - t0) / 60:.1f} min)", flush=True)
    return ZoneContext(zone=z, origin=origin, maps=maps, assoc=assoc, cal=cal, S=S, R=R, depths=depths,
                       W=W or 0, H=H or 0)


def rows_subset(rows: Dict[str, torch.Tensor], sel: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v[sel] for k, v in rows.items()}


def label_counts(cache_dir: str, zones: Sequence[str]) -> np.ndarray:
    cnt = np.zeros(11, np.int64)
    for z in zones:
        with np.load(os.path.join(cache_dir, z, "points.npz")) as p:
            lab = p["labels"]
        cnt += np.bincount(lab[lab != 255].astype(np.int64), minlength=11)
    return cnt


def class_weights_from_counts(cnt: np.ndarray, power: float = 0.5) -> np.ndarray:
    present = cnt > 0
    w = np.ones(11, np.float64)
    w[present] = (cnt[present].sum() / cnt[present]) ** power
    w[present] /= w[present].mean()
    return w.astype(np.float32)


# ----------------------------------------------------------------------------- trainer
class FusionTrainer:
    def __init__(self, backbone: FrozenBackbone, head: nn.Module, cfg: dict, work_dir: str,
                 class_w: Optional[np.ndarray] = None, device: str = "cuda"):
        self.backbone, self.head, self.cfg, self.work_dir, self.device = backbone, head.to(device), cfg, work_dir, device
        os.makedirs(work_dir, exist_ok=True)
        self.crit = SegLoss(class_w, ce_weight=1.0, lovasz_weight=cfg.get("lovasz", 1.0)).to(device)
        self.total_iters = cfg["passes"] * len(cfg["train_zones"]) * cfg["its_per_zone"]
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.01))
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, max_lr=cfg["lr"], total_steps=max(self.total_iters, 1),
                                                         pct_start=cfg.get("warmup", 0.05), div_factor=10, final_div_factor=100)
        self.it, self.best = 0, -1.0
        self.last_eval_it = -1                                  # iteration of the last validation run inside fit()
        self.cursor = {"pass": 0, "zone": 0, "done": 0}        # position inside the zone-major schedule
        self.rng = np.random.default_rng(cfg["seed"] + 7)
        self.aug_rng = np.random.default_rng(cfg["seed"] + 1)
        self.gen = torch.Generator(device=device); self.gen.manual_seed(cfg["seed"] + 11)
        cfg_path = os.path.join(work_dir, "config.json")
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w") as fh:
                json.dump(cfg, fh, indent=1)

    # ------------------------------------------------------------------ checkpoints / log
    def save(self, name="last.pth", extra: Optional[dict] = None):
        st = {"head": self.head.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
              "it": self.it, "best": self.best, "cursor": self.cursor, "cfg": self.cfg, "last_eval_it": self.last_eval_it,
              "rng": self.rng.bit_generator.state, "aug_rng": self.aug_rng.bit_generator.state,
              "gen": self.gen.get_state().cpu(), "torch_rng": torch.get_rng_state()}
        if extra:
            st.update(extra)
        path = os.path.join(self.work_dir, name)
        try:
            torch.save(st, path + ".tmp"); os.replace(path + ".tmp", path)
        except OSError as e:
            alt = os.path.join(tempfile.gettempdir(), "gridnet_ckpt_fusion"); os.makedirs(alt, exist_ok=True)
            torch.save(st, os.path.join(alt, name)); print(f"WARNING: could not write {path} ({e}); saved to {alt}", flush=True)

    def resume(self, name="last.pth", weights_only: bool = False) -> bool:
        path = os.path.join(self.work_dir, name)
        if not os.path.exists(path):
            return False
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.head.load_state_dict(st["head"])
        self.it, self.best = st["it"], st["best"]
        # checkpoints written before `last_eval_it` existed: a run that has a best score was evaluated up to here
        self.last_eval_it = st.get("last_eval_it", st["it"] if st["best"] > -1.0 else -1)
        if not weights_only:
            self.opt.load_state_dict(st["opt"]); self.sched.load_state_dict(st["sched"])
            self.cursor = st["cursor"]
            self.rng.bit_generator.state = st["rng"]; self.aug_rng.bit_generator.state = st["aug_rng"]
            self.gen.set_state(st["gen"])
            torch.set_rng_state(st["torch_rng"].cpu())
        print(f"[fusion] resumed {name} at it {self.it} (best {self.best:.2f}, cursor {self.cursor})", flush=True)
        return True

    def log(self, row: dict):
        path = os.path.join(self.work_dir, "log.csv")
        try:
            new = not os.path.exists(path)
            with open(path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row)); (w.writeheader() if new else None); w.writerow(row)
        except OSError as e:
            print(f"WARNING: log write failed ({e})", flush=True)

    # ------------------------------------------------------------------ one training step
    def step(self, ctx: ZoneContext, sampler: CropSampler) -> float:
        cfg = self.cfg
        z, idx = sampler.sample()
        c, f = augment(z.xyz[idx], z.feats[idx], self.aug_rng)
        batch = to_device(make_batch([(c, f, z.labels[idx])], cfg["grid"]), self.device)
        feat, logits = self.backbone(batch)
        rows = assoc_rows(ctx.assoc, idx, self.device)
        if cfg.get("pose_noise_t", 0) > 0 or cfg.get("pose_noise_r", 0) > 0:
            if self.rng.random() < cfg.get("aug_prob", 0.5):
                st = float(self.rng.uniform(0, cfg.get("pose_noise_t", 0)))          # metres
                sr = float(self.rng.uniform(0, cfg.get("pose_noise_r", 0)))          # degrees
                S2, R2 = perturb_cams(ctx.S, ctx.R, st, sr, self.gen)
                xyz_t = torch.from_numpy(z.xyz[idx]).to(self.device)
                rows = reproject_rows(rows, xyz_t, S2, R2, ctx.cal)      # visibility kept, as in the protocol's fast mode
        labels = batch["segment"]
        lab = torch.nonzero(labels != 255, as_tuple=False).squeeze(1)
        if len(lab) > cfg["head_points"]:
            lab = lab[torch.randperm(len(lab), device=self.device)[:cfg["head_points"]]]
        if len(lab) == 0:
            return float("nan")
        self.opt.zero_grad(set_to_none=True)
        chunk, total = cfg.get("head_chunk", 50_000), 0.0
        n_chunks = (len(lab) + chunk - 1) // chunk
        for s in range(0, len(lab), chunk):
            sel = lab[s:s + chunk]
            r = rows_subset(rows, sel)
            img_prob, img_feat = lift(ctx.maps, r, window=self.head.window)
            lg, ft, _ = drop_lidar(logits[sel], feat[sel], cfg.get("lidar_drop", 0.0), self.gen)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                out = self.head(lg, ft, img_prob, img_feat, r["desc"], r["mask"])
            loss = self.crit(out.float(), labels[sel]) / n_chunks
            if not torch.isfinite(loss):
                print(f"[fusion] non-finite loss at it {self.it}; chunk skipped", flush=True); continue
            loss.backward(); total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), cfg.get("clip", 1.0))
        self.opt.step(); self.sched.step()
        self.it += 1
        return total

    # ------------------------------------------------------------------ zone-major schedule
    def zone_order(self, pass_i: int) -> List[str]:
        names = list(self.cfg["train_zones"])
        perm = np.random.default_rng(self.cfg["seed"] + 1000 + pass_i).permutation(len(names))
        return [names[i] for i in perm]

    def fit(self, loader, val_names: Sequence[str], max_hours: float = 1e9, eval_fn=None) -> str:
        """`loader(zone_name, need_depths)` -> ZoneContext. Runs passes x zones x its_per_zone steps; evaluates on
        the val zones at the end of every pass (eval_fn(self) -> mIoU) and keeps the best head."""
        cfg, t0, hist = self.cfg, time.time(), []
        need_depths = False            # pose-noise augmentation re-projects the stored views and keeps their visibility
        self.head.train()
        for pass_i in range(self.cursor["pass"], cfg["passes"]):
            order = self.zone_order(pass_i)
            for zi in range(self.cursor["zone"] if pass_i == self.cursor["pass"] else 0, len(order)):
                done = self.cursor["done"] if (pass_i == self.cursor["pass"] and zi == self.cursor["zone"]) else 0
                if done >= cfg["its_per_zone"]:
                    continue
                ctx = loader(order[zi], need_depths)
                sampler = CropSampler([ctx.zone], radius=cfg["radius"], max_points=cfg["max_points"],
                                      rare_prob=cfg.get("rare_prob", 0.5), seed=cfg["seed"] + 100 * pass_i + zi)
                self.head.train()
                for j in range(done, cfg["its_per_zone"]):
                    loss = self.step(ctx, sampler)
                    hist.append(loss)
                    self.cursor = {"pass": pass_i, "zone": zi, "done": j + 1}
                    if self.it % cfg.get("log_every", 20) == 0:
                        el = (time.time() - t0) / 60
                        print(f"it {self.it}/{self.total_iters}  {order[zi]}  loss {np.nanmean(hist[-cfg.get('log_every', 20):]):.4f}  "
                              f"lr {self.sched.get_last_lr()[0]:.2e}  {el:.1f} min", flush=True)
                        self.log({"it": self.it, "zone": order[zi], "loss": float(np.nanmean(hist[-cfg.get('log_every', 20):])),
                                  "lr": self.sched.get_last_lr()[0], "min": el})
                    if self.it % cfg.get("ckpt_every", 100) == 0:
                        self.save("last.pth")
                    if (time.time() - t0) / 3600 > max_hours:
                        self.save("last.pth"); ctx.free(); return "time limit"
                ctx.free()
            self.cursor = {"pass": pass_i + 1, "zone": 0, "done": 0}
            self.save("last.pth")
            if eval_fn is not None and (pass_i + 1 == cfg["passes"] or cfg.get("eval_every_pass", True)):
                miou = eval_fn(self)
                self.last_eval_it = self.it
                self.log({"it": self.it, "pass": pass_i + 1, "val_mIoU": miou})
                if miou > self.best:
                    self.best = miou; self.save("best.pth"); print(f"[fusion] new best {miou:.2f}", flush=True)
                self.save("last.pth")
        return "done"

    # ------------------------------------------------------------------ inference on one zone
    @torch.no_grad()
    def predict_zone(self, ctx: ZoneContext, radius: Optional[float] = None, max_points: Optional[int] = None,
                     assoc: Optional[Association] = None, rows_fn=None) -> Tuple[np.ndarray, np.ndarray]:
        """Vote-averaged fused prediction and, for free, the LiDAR-only prediction of the frozen backbone on the
        same windows. `assoc` overrides the clean table (robustness protocol); `rows_fn(rows, idx)` can rewrite
        the gathered rows (pose-noise approximation). Runs under a fixed generator state (`fixed_torch_seed`), so
        repeated calls on the same weights return the same prediction."""
        with fixed_torch_seed(self.cfg.get("eval_seed", EVAL_SEED)):
            return self._predict_zone(ctx, radius, max_points, assoc, rows_fn)

    @torch.no_grad()
    def _predict_zone(self, ctx: ZoneContext, radius: Optional[float] = None, max_points: Optional[int] = None,
                      assoc: Optional[Association] = None, rows_fn=None) -> Tuple[np.ndarray, np.ndarray]:
        self.head.eval()
        cfg = self.cfg
        radius = radius or cfg["radius"]; max_points = max_points or cfg.get("eval_max_points", 900_000)
        z = ctx.zone; assoc = assoc or ctx.assoc
        votes = np.zeros((z.n, 11), np.float32); lvotes = np.zeros((z.n, 11), np.float32); counts = np.zeros(z.n, np.int32)
        chunk = cfg.get("head_chunk", 50_000)

        def run(idx):
            batch = to_device(make_batch([(z.xyz[idx], z.feats[idx], z.labels[idx])], cfg["grid"]), self.device)
            feat, logits = self.backbone(batch)
            rows = assoc_rows(assoc, idx, self.device)
            if rows_fn is not None:
                rows = rows_fn(rows, idx)
            out = torch.empty((len(idx), 11), device=self.device)
            for s in range(0, len(idx), chunk):
                sel = torch.arange(s, min(len(idx), s + chunk), device=self.device)
                r = rows_subset(rows, sel)
                img_prob, img_feat = lift(ctx.maps, r, window=self.head.window)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    out[sel] = self.head(logits[sel], feat[sel], img_prob, img_feat, r["desc"], r["mask"]).float()
            votes[idx] += torch.softmax(out, 1).cpu().numpy()
            lvotes[idx] += torch.softmax(logits, 1).cpu().numpy()
            counts[idx] += 1

        n_reg = 0
        for idx in eval_windows(z, radius=radius, max_points=max_points):
            run(idx); n_reg += 1
        n_extra = 0
        for idx in cover_remaining(z, counts, radius, max_points):
            run(idx); n_extra += 1
        if n_extra:
            print(f"[eval] {z.name}: {n_reg} windows + {n_extra} fill-in windows", flush=True)
        return votes.argmax(1).astype(np.uint8), lvotes.argmax(1).astype(np.uint8)

    def finish(self, loader, val_names: Sequence[str], save_dir: str) -> Tuple[dict, bool]:
        """Call after fit() returned "done". If the final weights were never validated (the session died during the
        last evaluation and the run was resumed), validate them now and update best.pth; then make eval/ describe
        best.pth without recomputing anything that is already there. Returns (summary of best, recomputed)."""
        if self.last_eval_it != self.it:
            summ = self.evaluate(loader, val_names, save_dir=save_dir, tag="pass")
            self.last_eval_it = self.it
            self.log({"it": self.it, "pass": self.cfg.get("passes"), "val_mIoU": summ["mIoU"]})
            if summ["mIoU"] > self.best:
                self.best = summ["mIoU"]; self.save("best.pth"); print(f"[fusion] new best {self.best:.2f}", flush=True)
            self.save("last.pth")
        return self.ensure_best_eval(loader, val_names, save_dir)

    def ensure_best_eval(self, loader, val_names: Sequence[str], save_dir: str) -> Tuple[dict, bool]:
        """Make <save_dir>/eval.json and the per-zone predictions describe best.pth. They already do when the last
        evaluation written there was run on the best checkpoint (same iteration, tag "pass"/"best"): then nothing
        is recomputed (a full evaluation of the six val zones costs ~20 min of GPU). Returns (summary, recomputed)."""
        best_path, ev_path = os.path.join(self.work_dir, "best.pth"), os.path.join(save_dir, "eval.json")
        best_it = torch.load(best_path, map_location="cpu", weights_only=False)["it"]
        last = None
        if os.path.exists(ev_path):
            with open(ev_path) as fh:
                last = json.load(fh)
        if last is not None and last.get("it") == best_it and last.get("tag") in ("pass", "best"):
            if last["tag"] != "best":
                last["tag"] = "best"
                with open(ev_path, "w") as fh:
                    json.dump(last, fh, indent=1)
            return last, False
        self.resume("best.pth", weights_only=True)
        return self.evaluate(loader, val_names, save_dir=save_dir, tag="best"), True

    def evaluate(self, loader, val_names: Sequence[str], save_dir: Optional[str] = None, tag: str = "") -> dict:
        """Fused and LiDAR-only metrics on the val zones; writes <save_dir>/eval.json (plot_eval.py format) and
        per-zone predictions. Returns the summary dict."""
        meter, lmeter, t0 = ConfusionMeter(), ConfusionMeter(), time.time()
        per_zone = {}
        for name in val_names:
            ctx = loader(name, False)
            pred, lpred = self.predict_zone(ctx)
            zm = ConfusionMeter(); zm.add(ctx.zone.labels, pred); meter.m += zm.m
            lz = ConfusionMeter(); lz.add(ctx.zone.labels, lpred); lmeter.m += lz.m
            per_zone[name] = {"mIoU": 100 * zm.miou(present_only=True), "OA": 100 * zm.oa(),
                              "per_class": {n: (None if np.isnan(v) else float(100 * v)) for n, v in zip(CLASS_NAMES, zm.iou())},
                              "lidar_only_mIoU": 100 * lz.miou(present_only=True)}
            print(f"[eval] {name}: fused mIoU {per_zone[name]['mIoU']:.2f}  OA {100 * zm.oa():.2f}  "
                  f"(LiDAR-only {per_zone[name]['lidar_only_mIoU']:.2f})", flush=True)
            if save_dir:
                os.makedirs(os.path.join(save_dir, name), exist_ok=True)
                np.save(os.path.join(save_dir, name, "fused_pred.npy"), pred)
            ctx.free()
        ious = meter.iou()
        summary = {"model": self.cfg.get("head"), "tag": tag, "it": self.it, "mIoU": 100 * meter.miou(), "OA": 100 * meter.oa(),
                   "per_class": {n: (None if np.isnan(v) else float(100 * v)) for n, v in zip(CLASS_NAMES, ious)},
                   "per_zone": per_zone, "confusion": meter.m.tolist(), "class_names": CLASS_NAMES,
                   "lidar_only": {"mIoU": 100 * lmeter.miou(), "OA": 100 * lmeter.oa(),
                                  "per_class": {n: (None if np.isnan(v) else float(100 * v)) for n, v in zip(CLASS_NAMES, lmeter.iou())}},
                   "minutes": (time.time() - t0) / 60}
        print(f"[eval] it {self.it}  fused mIoU {summary['mIoU']:.2f}  OA {summary['OA']:.2f}   "
              f"LiDAR-only mIoU {summary['lidar_only']['mIoU']:.2f}  ({summary['minutes']:.1f} min)", flush=True)
        print("       " + "  ".join(f"{n} {100 * v:.1f}" for n, v in zip(CLASS_NAMES, ious) if not np.isnan(v)), flush=True)
        if save_dir:
            with open(os.path.join(save_dir, "eval.json"), "w") as fh:
                json.dump(summary, fh, indent=1)
        return summary
