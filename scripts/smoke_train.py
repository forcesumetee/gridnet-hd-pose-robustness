"""Smoke test: LitePT-S + linear head trained for a few hundred iterations on crops of one zone.

  python smoke_train.py --smoke /content/smoke.npz --litept /content/LitePT --ckpt-dir out --iters 300

Purpose (gate G1): prove the environment works, that loss decreases on our data, and measure
s/it and peak VRAM. Not an accuracy experiment.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

CLASS_NAMES = ["pylon", "conductor", "structural", "insulator", "high_veg", "low_veg", "herb",
               "rock_soil", "road", "water", "building"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--litept", default="/content/LitePT")
    ap.add_argument("--ckpt-dir", default="out")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--max-points", type=int, default=160_000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="load litept_smoke.pth from --ckpt-dir and only evaluate")
    a = ap.parse_args()

    sys.path.insert(0, a.litept)
    from litept.model import LitePT  # noqa: E402  (imports flash_attn, spconv, torch_scatter)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = np.load(a.smoke)
    XYZ, LAB, FEATS = d["xyz"].astype(np.float32), d["labels"], d["feats"].astype(np.float32)
    GRID = float(d["grid"])
    print(f"zone {d['zone']}: {len(LAB):,} points at {GRID} m, feats {list(d['feat_names'])}", flush=True)

    axis = int(np.argmax(XYZ.max(0) - XYZ.min(0)))
    cut = np.quantile(XYZ[:, axis], 0.7)
    train_mask = XYZ[:, axis] <= cut
    print(f"train points {train_mask.sum():,} | val points {(~train_mask).sum():,}", flush=True)
    tree_xy = cKDTree(XYZ[:, :2])

    def sample_crop(mask, rare_prob=0.7):
        pool = np.where(mask)[0]
        rare = pool[np.isin(LAB[pool], (0, 1, 2, 3))]
        c = rng.choice(rare) if (len(rare) and rng.random() < rare_prob) else rng.choice(pool)
        idx = np.array(tree_xy.query_ball_point(XYZ[c, :2], a.radius))
        idx = idx[mask[idx]]
        if len(idx) > a.max_points:
            idx = rng.choice(idx, a.max_points, replace=False)
        return idx

    def make_batch(idx_list, augment=True):
        coords, feats, segs, offsets, n = [], [], [], [], 0
        for idx in idx_list:
            c = XYZ[idx] - XYZ[idx].mean(0)
            if augment:
                th = rng.uniform(-np.pi, np.pi); s = rng.uniform(0.9, 1.1)
                Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
                c = (c @ Rz.T) * s
            coords.append(c); feats.append(np.concatenate([c, FEATS[idx]], 1)); segs.append(LAB[idx])
            n += len(idx); offsets.append(n)
        coord = torch.from_numpy(np.concatenate(coords)).float()
        grid_coord = torch.div(coord - coord.min(0).values, GRID, rounding_mode="trunc").int()
        return dict(coord=coord, grid_coord=grid_coord, feat=torch.from_numpy(np.concatenate(feats)).float(),
                    segment=torch.from_numpy(np.concatenate(segs)).long(), offset=torch.tensor(offsets))

    class SegModel(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.backbone = LitePT(in_channels=in_ch)   # LitePT-S defaults, decoder out = 72 ch
            self.head = nn.Linear(72, 11)

        def forward(self, batch):
            return self.head(self.backbone(batch).feat)

    in_ch = 3 + FEATS.shape[1]
    model = SegModel(in_ch).cuda()
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)

    # forward check
    batch = {k: v.cuda() for k, v in make_batch([sample_crop(train_mask)]).items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        t0 = time.time(); logits = model(batch); torch.cuda.synchronize()
    print(f"forward ok: logits {tuple(logits.shape)} in {time.time() - t0:.2f} s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)

    if a.eval_only:
        ck = torch.load(os.path.join(a.ckpt_dir, "litept_smoke.pth"), map_location="cuda")
        model.load_state_dict(ck["model"]); print("loaded checkpoint", flush=True)
    # training
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.005)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda')
    cnt = np.bincount(LAB[train_mask][LAB[train_mask] != 255], minlength=11).astype(np.float64) + 1
    w = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float32); w = (w / w.mean()).cuda()
    crit = nn.CrossEntropyLoss(weight=w, ignore_index=255)
    model.train(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); hist = []
    for it in range(1, 0 if a.eval_only else a.iters + 1):
        batch = {k: v.cuda(non_blocking=True) for k, v in
                 make_batch([sample_crop(train_mask) for _ in range(a.batch)]).items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(model(batch), batch["segment"])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        hist.append(loss.item())
        if it % 25 == 0 or it == 1:
            print(f"it {it:4d}  loss {np.mean(hist[-25:]):.3f}  {(time.time() - t0) / it:.2f} s/it  "
                  f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    if not a.eval_only:
        torch.save({"model": model.state_dict(), "iters": a.iters, "grid": GRID, "in_ch": in_ch},
               os.path.join(a.ckpt_dir, "litept_smoke.pth"))

    # quick eval on the held-out part of the zone. fp32 on purpose: spconv 2.3 implicit-GEMM tuner
    # fails in eval mode under fp16 autocast for channel counts not divisible by 8 (LitePT stage 0 = 36)
    model.eval(); conf = np.zeros((11, 11), np.int64)
    with torch.no_grad():
        for _ in range(12):
            batch = {k: v.cuda() for k, v in make_batch([sample_crop(~train_mask)], augment=False).items()}
            pred = model(batch).argmax(1).cpu().numpy(); gt = batch["segment"].cpu().numpy()
            m = gt != 255
            np.add.at(conf, (gt[m], pred[m]), 1)
    inter = np.diag(conf); union = conf.sum(0) + conf.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    for c, name in enumerate(CLASS_NAMES):
        if union[c] > 0:
            print(f"{name:<12} IoU {100 * iou[c]:5.1f}%   gt points {conf.sum(1)[c]:,}")
    print(f"mIoU over present classes: {100 * np.nanmean(iou):.1f}%")
    print(f"total time {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
