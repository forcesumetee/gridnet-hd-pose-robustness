"""Resume-safe training / evaluation engine for LitePT-S on cached GridNet-HD zones.

Designed for pre-emptible sessions (Colab): checkpoints every `ckpt_every` iterations to a
directory on Drive, restores model/optimizer/scheduler/scaler/RNG, and stops cleanly when
`max_hours` is reached so the next session continues from the same point.
"""
import csv
import json
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .data import ConfusionMeter, CropSampler, Zone, augment, cover_remaining, eval_windows, make_batch
from .losses import SegLoss

CLASS_NAMES = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb",
               "rock_soil", "road", "water", "building"]


def build_model(litept_root: str, in_channels: int, num_classes: int = 11, drop_path: float = 0.3) -> nn.Module:
    sys.path.insert(0, litept_root)
    from litept.model import LitePT  # noqa: E402

    class SegModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = LitePT(in_channels=in_channels, drop_path=drop_path)   # LitePT-S defaults
            self.head = nn.Sequential(nn.LayerNorm(72), nn.Linear(72, num_classes))

        def forward(self, batch):
            return self.head(self.backbone(batch).feat)

    return SegModel()


def to_device(b: Dict[str, np.ndarray], device="cuda") -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in b.items():
        t = torch.from_numpy(v)
        if k == "grid_coord":
            t = t.int()
        out[k] = t.to(device, non_blocking=True)
    return out


class Trainer:
    def __init__(self, model: nn.Module, train_zones: Sequence[Zone], val_zones: Sequence[Zone], cfg: dict,
                 ckpt_dir: str, class_w: Optional[np.ndarray] = None):
        self.model = model.cuda()
        self.cfg = cfg
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self.sampler = CropSampler(train_zones, radius=cfg["radius"], max_points=cfg["max_points"],
                                   rare_prob=cfg["rare_prob"], seed=cfg["seed"])
        self.val_zones = list(val_zones)
        self.aug_rng = np.random.default_rng(cfg["seed"] + 1)
        self.crit = SegLoss(class_w, ce_weight=1.0, lovasz_weight=cfg["lovasz"]).cuda()
        # LitePT/PTv3 recipe: transformer/conv *blocks* train at 0.1x the base LR (param_dicts keyword
        # "block"); embedding, pooling and head at the base LR. One LR for everything (2e-3) diverged
        # to NaN at it ~2350 on GridNet-HD.
        mult = cfg.get("block_lr_mult", 0.1)
        blocks = [p for n, p in self.model.named_parameters() if "block" in n]
        others = [p for n, p in self.model.named_parameters() if "block" not in n]
        groups = [{"params": others, "lr": cfg["lr"]}, {"params": blocks, "lr": cfg["lr"] * mult}]
        print(f"[engine] param groups: {sum(p.numel() for p in others) / 1e6:.2f} M @ lr {cfg['lr']:.1e}, "
              f"{sum(p.numel() for p in blocks) / 1e6:.2f} M blocks @ lr {cfg['lr'] * mult:.1e}", flush=True)
        self.opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["wd"])
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, max_lr=[cfg["lr"], cfg["lr"] * mult],
                                                         total_steps=cfg["iters"], pct_start=cfg["warmup"],
                                                         div_factor=10, final_div_factor=100)
        self.scaler = torch.amp.GradScaler("cuda")
        self.it = 0
        self.best = -1.0
        self.log_path = os.path.join(ckpt_dir, "log.csv")
        cfg_path = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(cfg_path):          # same ckpt_dir = same run; eval-only must not overwrite it
            with open(cfg_path, "w") as fh:
                json.dump(cfg, fh, indent=1)

    # ------------------------------------------------------------------ checkpointing
    def save(self, name="last.pth"):
        state = {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                 "scaler": self.scaler.state_dict(), "it": self.it, "best": self.best,
                 "sampler": self.sampler.state(), "aug_rng": self.aug_rng.bit_generator.state,
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "cfg": self.cfg}
        tmp = os.path.join(self.ckpt_dir, name + ".tmp")
        try:
            torch.save(state, tmp)
            os.replace(tmp, os.path.join(self.ckpt_dir, name))
        except OSError as e:
            # Drive mount hiccup (I/O error, transport endpoint not connected): keep training,
            # park the checkpoint on local disk so nothing is lost while the session lives
            local = os.path.join(tempfile.gettempdir(), "gridnet_ckpt")
            os.makedirs(local, exist_ok=True)
            torch.save(state, os.path.join(local, name))
            print(f"[engine] WARNING: could not write {name} to {self.ckpt_dir} ({e}); "
                  f"saved to {local}/{name} instead — copy it to Drive when the mount is back", flush=True)

    def resume(self, name="last.pth") -> bool:
        path = os.path.join(self.ckpt_dir, name)
        if not os.path.exists(path):
            return False
        # map_location="cpu": RNG states must stay CPU ByteTensors (map_location="cuda" moved them to
        # the GPU and torch.set_rng_state raised). Model/optimizer state is moved to the GPU by
        # load_state_dict itself.
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(st["model"]); self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"]); self.scaler.load_state_dict(st["scaler"])
        self.it, self.best = st["it"], st["best"]
        self.sampler.load_state(st["sampler"]); self.aug_rng.bit_generator.state = st["aug_rng"]
        torch.set_rng_state(st["torch_rng"].cpu()); torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        print(f"[engine] resumed from {path} at it {self.it} (best {self.best:.2f})", flush=True)
        return True

    def log(self, row: dict):
        try:
            new = not os.path.exists(self.log_path)
            with open(self.log_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow(row)
        except OSError as e:          # never let a Drive hiccup kill the run over a log line
            print(f"[engine] WARNING: log write failed ({e})", flush=True)

    # ------------------------------------------------------------------ training
    def train_batch(self):
        items = []
        for _ in range(self.cfg["batch"]):
            z, idx = self.sampler.sample()
            c, f = augment(z.xyz[idx], z.feats[idx], self.aug_rng)
            items.append((c, f, z.labels[idx]))
        return to_device(make_batch(items, self.cfg["grid"]))

    def fit(self, max_hours: float = 1e9):
        cfg = self.cfg
        t_start = time.time(); t_last = t_start; hist: List[float] = []
        oom_streak = 0; nonfinite_streak = 0; n_nonfinite = 0
        self.model.train()
        while self.it < cfg["iters"]:
            batch = self.train_batch()
            n_pts = int(batch["coord"].shape[0])
            if self.it == 0 and not hist:
                print(f"[engine] first batch: {n_pts:,} points ({cfg['batch']} crops, cap {cfg['max_points']:,}/crop)", flush=True)
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = self.model(batch)
                loss = self.crit(logits, batch["segment"])
                if not torch.isfinite(loss):
                    # fp16 overflow or a bad batch: never backprop a NaN. GradScaler only guards the
                    # gradients, not a non-finite forward, so this is the line that keeps weights sane.
                    nonfinite_streak += 1; n_nonfinite += 1
                    del logits, loss, batch; self.opt.zero_grad(set_to_none=True)
                    if nonfinite_streak <= 3 or nonfinite_streak % 10 == 0:
                        print(f"[engine] non-finite loss at it {self.it} ({nonfinite_streak} in a row, {n_nonfinite} total) — batch skipped", flush=True)
                    if nonfinite_streak >= 20:
                        self.save("last_diverged.pth")
                        raise RuntimeError("loss non-finite 20 iterations in a row: training diverged — lower --lr (and check --block-lr-mult)")
                    continue
                nonfinite_streak = 0
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                # a rare very dense crop: drop the batch instead of killing a multi-hour session
                oom_streak += 1
                del batch; logits = loss = None
                self.opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                print(f"[engine] CUDA OOM at it {self.it} on a {n_pts:,}-point batch — skipped "
                      f"({oom_streak} in a row). Lower --max-points if this repeats.", flush=True)
                if oom_streak >= 5:
                    raise
                continue
            oom_streak = 0
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg["clip"])
            self.scaler.step(self.opt); self.scaler.update(); self.sched.step()
            self.it += 1; hist.append(loss.item())
            if self.it % cfg["log_every"] == 0:
                now = time.time()
                print(f"it {self.it:6d}/{cfg['iters']}  loss {np.mean(hist[-cfg['log_every']:]):.3f}  "
                      f"{(now - t_last) / cfg['log_every']:.2f} s/it  lr {self.sched.get_last_lr()[0]:.2e}  "
                      f"VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
                      + (f"  skipped {n_nonfinite}" if n_nonfinite else ""), flush=True)
                self.log({"it": self.it, "loss": float(np.mean(hist[-cfg["log_every"]:])),
                          "lr": self.sched.get_last_lr()[0], "val_miou": "", "time": now})
                t_last = now
            if self.it % cfg["eval_every"] == 0 or self.it == cfg["iters"]:
                miou, ious, oa = self.evaluate()
                self.log({"it": self.it, "loss": "", "lr": self.sched.get_last_lr()[0], "val_miou": miou, "time": time.time()})
                if miou > self.best:
                    self.best = miou; self.save("best.pth")
                self.model.train()
            if self.it % cfg["ckpt_every"] == 0:
                self.save("last.pth")
            if (time.time() - t_start) / 3600 > max_hours:
                self.save("last.pth")
                print(f"[engine] max_hours reached at it {self.it}; checkpoint saved, exit for resume", flush=True)
                return "paused"
        self.save("last.pth")
        return "done"

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def predict_zone(self, zone: Zone, radius: Optional[float] = None, max_points: Optional[int] = None) -> np.ndarray:
        """Vote-averaged softmax over overlapping windows; returns argmax per point (uint8).
        Windows above `max_points` are shrunk around their centre, and a second pass centred on
        still-unpredicted points guarantees that every point receives at least one vote."""
        self.model.eval()
        radius = radius or self.cfg["radius"]
        max_points = max_points or self.cfg.get("eval_max_points", 900_000)
        votes = np.zeros((zone.n, 11), np.float32)
        counts = np.zeros(zone.n, np.int32)

        def run(idx):
            b = to_device(make_batch([(zone.xyz[idx], zone.feats[idx], zone.labels[idx])], self.cfg["grid"]))
            logits = self.model(b).float()          # fp32: spconv's fp16 tuner fails in eval mode
            votes[idx] += torch.softmax(logits, 1).cpu().numpy()
            counts[idx] += 1

        n_regular = 0
        for idx in eval_windows(zone, radius=radius, max_points=max_points):
            run(idx); n_regular += 1
        n_extra = 0
        for idx in cover_remaining(zone, counts, radius, max_points):
            run(idx); n_extra += 1
        if n_extra:
            print(f"[eval] {zone.name}: {n_regular} windows + {n_extra} fill-in windows for dense areas", flush=True)
        torch.cuda.empty_cache()
        return votes.argmax(1).astype(np.uint8)

    def evaluate(self, save_dir: Optional[str] = None) -> tuple:
        """Predict every val zone; returns (mIoU %, per-class IoU, OA %). Also fills
        `self.confusion` (11x11 counts over all val zones, rows = ground truth) and
        `self.per_zone` ({zone: {mIoU, OA, iou}}); with `save_dir`, the per-point predictions of
        each zone are written to <save_dir>/<zone>_pred.npy (uint8, cache point order)."""
        meter = ConfusionMeter()
        self.per_zone = {}
        t0 = time.time()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        for z in self.val_zones:
            pred = self.predict_zone(z)
            zm = ConfusionMeter()
            zm.add(z.labels, pred)
            meter.m += zm.m
            self.per_zone[z.name] = {"mIoU": 100 * zm.miou(present_only=True), "OA": 100 * zm.oa(), "iou": zm.iou()}
            print(f"[eval]   {z.name}: mIoU {100 * zm.miou(present_only=True):.2f} (classes present)  OA {100 * zm.oa():.2f}", flush=True)
            if save_dir:
                np.save(os.path.join(save_dir, f"{z.name}_pred.npy"), pred)
        self.confusion = meter.m.copy()
        ious = meter.iou()
        print(f"[eval] it {self.it}  mIoU {100 * meter.miou():.2f}  OA {100 * meter.oa():.2f}  ({(time.time() - t0) / 60:.1f} min)")
        print("       " + "  ".join(f"{n} {100 * v:.1f}" for n, v in zip(CLASS_NAMES, ious) if not np.isnan(v)), flush=True)
        return 100 * meter.miou(), ious, 100 * meter.oa()
