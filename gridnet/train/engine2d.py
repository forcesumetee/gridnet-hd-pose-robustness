"""Training / evaluation / cache export for the 2D branch (UPerNet). Resume-safe, Colab-friendly."""
import csv
import json
import os
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import ConfusionMeter
from .data2d import CropDataset, load_image_tensor
from .losses import lovasz_softmax_flat
from .upernet import UPerNet, sliding_window

CLASS_NAMES = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb", "rock_soil", "road", "water", "building"]


def seg_loss_2d(logits: torch.Tensor, target: torch.Tensor, class_w: Optional[torch.Tensor],
                lovasz_weight: float = 1.0, lovasz_pixels: int = 100_000) -> torch.Tensor:
    """Weighted CE on all pixels (label 255 ignored) + Lovász-softmax on a random subset of labelled pixels."""
    logits = logits.float()
    loss = F.cross_entropy(logits, target, weight=class_w, ignore_index=255)
    if lovasz_weight > 0:
        flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        t = target.reshape(-1)
        keep = torch.nonzero(t != 255, as_tuple=False).squeeze(1)
        if len(keep) > lovasz_pixels:
            keep = keep[torch.randperm(len(keep), device=keep.device)[:lovasz_pixels]]
        if len(keep):
            loss = loss + lovasz_weight * lovasz_softmax_flat(F.softmax(flat[keep], 1), t[keep])
    return loss


class Trainer2D:
    def __init__(self, model: UPerNet, train_ds: Optional[CropDataset], val_ds: Optional[CropDataset], cfg: dict,
                 ckpt_dir: str, class_w: Optional[np.ndarray] = None, device: str = "cuda"):
        self.model, self.cfg, self.ckpt_dir, self.device = model.to(device), cfg, ckpt_dir, device
        self.train_ds, self.val_ds = train_ds, val_ds
        self.class_w = None if class_w is None else torch.as_tensor(class_w, dtype=torch.float32, device=device)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.it, self.best = 0, -1.0
        if train_ds is not None:
            groups = model.param_groups(cfg["lr"], cfg.get("encoder_lr_mult", 0.1), cfg.get("wd", 0.01))
            self.opt = torch.optim.AdamW(groups)
            self.sched = torch.optim.lr_scheduler.OneCycleLR(
                self.opt, max_lr=[g["lr"] for g in groups], total_steps=cfg["iters"],
                pct_start=cfg.get("warmup", 0.05), div_factor=10, final_div_factor=100, anneal_strategy="cos")
            self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True) and device == "cuda")
            self.loader = torch.utils.data.DataLoader(
                train_ds, batch_size=cfg["batch"], shuffle=True, num_workers=cfg.get("workers", 4), drop_last=True,
                pin_memory=device == "cuda", persistent_workers=cfg.get("workers", 4) > 0)
        cfg_path = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w") as fh:
                json.dump(cfg, fh, indent=1)

    # ------------------------------------------------------------------ checkpoints
    def save(self, name="last.pth"):
        st = {"model": self.model.state_dict(), "it": self.it, "best": self.best, "cfg": self.cfg}
        if self.train_ds is not None:
            st.update(opt=self.opt.state_dict(), sched=self.sched.state_dict(), scaler=self.scaler.state_dict())
        path = os.path.join(self.ckpt_dir, name)
        try:
            torch.save(st, path + ".tmp"); os.replace(path + ".tmp", path)
        except OSError as e:                      # Drive hiccup: keep the run alive, save locally
            alt = os.path.join(tempfile.gettempdir(), "gridnet_ckpt2d"); os.makedirs(alt, exist_ok=True)
            torch.save(st, os.path.join(alt, name)); print(f"WARNING: could not write {path} ({e}); saved to {alt}", flush=True)

    def resume(self, name="last.pth") -> bool:
        path = os.path.join(self.ckpt_dir, name)
        if not os.path.exists(path):
            return False
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(st["model"])
        self.it, self.best = st["it"], st["best"]
        if self.train_ds is not None and "opt" in st:
            self.opt.load_state_dict(st["opt"]); self.sched.load_state_dict(st["sched"]); self.scaler.load_state_dict(st["scaler"])
        print(f"[engine2d] resumed {name} at it {self.it} (best {self.best:.2f})", flush=True)
        return True

    def log(self, row: dict):
        path = os.path.join(self.ckpt_dir, "log.csv")
        try:
            new = not os.path.exists(path)
            with open(path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row)); (w.writeheader() if new else None); w.writerow(row)
        except OSError as e:
            print(f"WARNING: log write failed ({e})", flush=True)

    # ------------------------------------------------------------------ training
    def fit(self, max_hours: float = 1e9) -> str:
        cfg, t0 = self.cfg, time.time()
        self.model.train()
        loader_it = iter(self.loader)
        run_loss, n_loss = 0.0, 0
        while self.it < cfg["iters"]:
            if (time.time() - t0) / 3600 > max_hours:
                self.save("last.pth"); return "time limit"
            try:
                x, y = next(loader_it)
            except StopIteration:
                loader_it = iter(self.loader); x, y = next(loader_it)
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.scaler.is_enabled()):
                logits = self.model(x)                                            # (B, C, H/4, W/4)
            # supervise at full resolution (bilinear x4) so 1-2 px cables are not lost by downsampling the target
            logits = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = seg_loss_2d(logits, y, self.class_w, cfg.get("lovasz", 1.0))
            if not torch.isfinite(loss):
                print(f"[engine2d] non-finite loss at it {self.it}, batch skipped", flush=True)
                self.opt.zero_grad(set_to_none=True); continue
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.get("clip", 1.0))
            self.scaler.step(self.opt); self.scaler.update(); self.sched.step()
            self.it += 1; run_loss += loss.item(); n_loss += 1
            if self.it % cfg.get("log_every", 20) == 0:
                el = (time.time() - t0) / 60
                print(f"it {self.it}/{cfg['iters']}  loss {run_loss / n_loss:.4f}  lr {self.sched.get_last_lr()[-1]:.2e}  {el:.1f} min", flush=True)
                self.log({"it": self.it, "loss": run_loss / n_loss, "lr": self.sched.get_last_lr()[-1], "min": el})
                run_loss, n_loss = 0.0, 0
            if self.it % cfg.get("ckpt_every", 250) == 0:
                self.save("last.pth")
            if self.val_ds is not None and self.it % cfg.get("eval_every", 1000) == 0:
                miou, ious, oa = self.evaluate()
                self.log({"it": self.it, "val_mIoU": miou, "val_OA": oa})
                if miou > self.best:
                    self.best = miou; self.save("best.pth"); print(f"[engine2d] new best {miou:.2f}", flush=True)
                self.model.train()
        self.save("last.pth")
        return "done"

    # ------------------------------------------------------------------ evaluation on fixed val crops
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, np.ndarray, float]:
        self.model.eval()
        meter = ConfusionMeter()
        loader = torch.utils.data.DataLoader(self.val_ds, batch_size=self.cfg["batch"], num_workers=self.cfg.get("workers", 4))
        t0 = time.time()
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                logits = self.model(x)
            pred = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False).argmax(1).cpu().numpy()
            meter.add(y.numpy().astype(np.uint8), pred.astype(np.uint8))
        self.confusion = meter.m.copy()
        ious = meter.iou()
        print(f"[eval2d] it {self.it}  mIoU {100 * meter.miou():.2f}  OA {100 * meter.oa():.2f}  ({(time.time() - t0) / 60:.1f} min)")
        print("         " + "  ".join(f"{n} {100 * v:.1f}" for n, v in zip(CLASS_NAMES, ious) if not np.isnan(v)), flush=True)
        return 100 * meter.miou(), ious, 100 * meter.oa()

    # ------------------------------------------------------------------ cache export
    @torch.no_grad()
    def export_cache(self, root: str, zones: Sequence[str], out_dir: str, crop: int = 1024, stride: int = 768,
                     batch: int = 4, feat_pool: int = 2, limit: Optional[int] = None, compact: bool = True) -> int:
        """Per image: softmax probabilities at 1/4 res + features average-pooled to 1/8 res, written to
        <out_dir>/<zone>/<image_id>.npz. `compact` (default) stores prob as uint8 (x255) and feat as int8 with a
        per-channel scale `feat_scale` (~24 MB/image); otherwise both are fp16 (~50 MB/image, the format of the
        first val-zone caches). Readers (`gridnet.train.maps2d.load_maps`) accept both. Existing files are
        skipped (resumable). Returns #written."""
        from ..io import list_image_ids
        self.model.eval()
        n_done, t0 = 0, time.time()
        for z in zones:
            ids = list_image_ids(os.path.join(root, z))
            if limit:
                ids = ids[:limit]
            zdir = os.path.join(out_dir, z); os.makedirs(zdir, exist_ok=True)
            todo = [i for i in ids if not os.path.exists(os.path.join(zdir, i + ".npz"))]
            print(f"[export] {z}: {len(ids)} images, {len(todo)} to do", flush=True)
            for k, i in enumerate(todo):
                img = load_image_tensor(root, z, i).to(self.device)
                logits, feat = sliding_window(self.model, img, crop=crop, stride=stride, batch=batch)
                prob = torch.softmax(logits, 0)
                feat = F.avg_pool2d(feat[None], feat_pool)[0] if feat_pool > 1 else feat
                tmp = os.path.join(zdir, i + ".npz.tmp.npz")
                if compact:
                    prob = prob.mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    scale = feat.float().abs().amax(dim=(1, 2)).clamp(min=1e-6) / 127.0           # (C,)
                    feat_i8 = (feat.float() / scale[:, None, None]).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
                    np.savez(tmp, prob=prob, feat=feat_i8, feat_scale=scale.cpu().numpy().astype(np.float32))
                else:
                    prob = prob.half().cpu().numpy(); feat = feat.half().cpu().numpy()
                    np.savez(tmp, prob=prob, feat=feat)
                os.replace(tmp, os.path.join(zdir, i + ".npz"))
                n_done += 1
                if (k + 1) % 20 == 0 or k + 1 == len(todo):
                    rate = (time.time() - t0) / n_done
                    print(f"[export] {z} {k + 1}/{len(todo)}  {rate:.1f} s/img  prob {tuple(prob.shape)} feat {tuple(feat.shape)}", flush=True)
        return n_done
