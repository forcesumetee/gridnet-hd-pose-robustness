"""Fast head training on precomputed frozen-backbone features (a "feature bank").

With both backbones frozen a head only ever sees (a) the LiDAR feature / logits of a point and (b) the image
evidence read through the point's association rows. `FusionTrainer` recomputes (a) with a backbone forward pass in
every iteration (0.6 s/it; 1.75 h per head with two validations, measured on concat_v1). Here (a) is computed ONCE
per zone for a class-stratified subset of the points - the same `subset.npz` the robustness protocol uses
(window-averaged features, fixed generator state) - so a training iteration is a gather from the 2D maps plus a
head forward/backward, and any number of head variants (designs, seeds, augmentation settings) are trained side by
side on shared batches while one zone's maps are resident on the GPU. Validation uses the banks of the validation
zones with inverse-sampling-rate weights, i.e. an unbiased estimate of the full-zone confusion matrix, and equals
the protocol's level-0 evaluation of the same head.

Checkpoints are written per head as <out_root>/<name>/best.pth with the keys FusionTrainer uses ("head", "cfg",
"it", "best"), so `ProtocolModels.from_checkpoints` and `train_fusion.py --eval-only` (full-zone evaluation) read
them unchanged.

Batch sampling: a share `natural_frac` of each batch is drawn with probability proportional to the inverse sampling
rate (natural class frequencies), the rest uniformly over the stratified bank (rare classes boosted). Default 1.0:
in the first lab session (2026-11-13) the late head trained with natural batches reproduced the crop-trained late_v1
(81.8 vs 81.9 mIoU) while natural_frac = 0.5 cost 1.4 mIoU (pylon, building, herbaceous, rock/soil).

Inference differs from `FusionTrainer.predict_zone` in one respect: there the head is applied to the LiDAR feature
of every evaluation window and the class probabilities are averaged over the windows that contain a point; here
the LiDAR feature itself is averaged over those windows and the head is applied once. `train_fusion.py --eval-only`
gives the window-vote evaluation of a bank-trained head; `add_references` gives the bank evaluation of a
crop-trained one.
"""
import copy
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..camera import load_zone_cameras
from .assoc_gpu import zone_cams
from .engine import CLASS_NAMES
from .fusion import ZoneMaps, build_head, drop_lidar, lift, perturb_cams, reproject_rows, uniform_weights
from .losses import SegLoss
from .protocol import IMAGEVOTE_FALLBACK, load_subset, subset_rows, weighted_confusion, window_slice


# ----------------------------------------------------------------------------- what to train
@dataclass
class HeadSpec:
    name: str                       # run name -> <out_root>/<name>/
    head: str = "late"              # late | concat | window
    k: int = 3                      # window size (window head)
    lidar_input: str = "feat"       # concat / window: "feat" (72-d decoder feature) | "prob" (class probabilities)
    image_input: str = "feat"       # concat: "feat" (image feature + probabilities) | "prob" (probabilities only)
    lidar_drop: float = 0.0         # LiDAR modality dropout during training (fraction of points)
    pose_t: float = 0.0             # pose-noise augmentation: sigma_t ~ U(0, pose_t) metres
    pose_r: float = 0.0             # sigma_r ~ U(0, pose_r) degrees
    aug_prob: float = 0.5           # share of iterations that are perturbed when pose noise is on
    seed: int = 0                   # head initialisation, dropout and batch sampling
    natural_frac: float = 1.0       # share of a batch drawn at natural class frequencies (rest: uniform over the bank)
    lr: float = 1e-3
    wd: float = 0.01
    uniform_views: bool = False     # late head only

    def build(self) -> nn.Module:
        kw = {"lidar_input": self.lidar_input, "image_input": self.image_input}
        if self.head == "late" and self.uniform_views:
            kw["uniform_views"] = True
        return build_head(self.head, k=self.k, **kw)

    def group(self):                # heads of one group share batches and gathered image evidence
        return (self.seed, self.pose_t, self.pose_r, self.aug_prob, self.natural_frac)


_CASTS = {"k": int, "seed": int, "lidar_drop": float, "pose_t": float, "pose_r": float, "aug_prob": float, "lr": float,
          "wd": float, "natural_frac": float, "lidar_input": str, "image_input": str,
          "uniform_views": lambda v: str(v).lower() in ("1", "true", "yes")}
# what makes a trained head "the same experiment": a folder written with other values is never overwritten
_IDENTITY = ("train_zones", "passes", "its_per_zone", "batch_points", "warmup", "clip", "lovasz", "class_weight_power", "tau",
             "seed_order", "head", "k", "lidar_input", "image_input", "uniform_views", "lidar_drop", "pose_noise_t",
             "pose_noise_r", "aug_prob", "natural_frac", "seed", "lr", "wd")


def parse_specs(text: str) -> List[HeadSpec]:
    """'name:head[:key=value,key=value]' entries separated by whitespace or ';'.
    Example: 'late_b:late  concat_ld50:concat:lidar_drop=0.5  win3p:window:k=3,lidar_input=prob  late_s1:late:seed=1'"""
    specs = []
    for entry in text.replace(";", " ").split():
        parts = entry.split(":")
        if len(parts) < 2 or parts[1] not in ("late", "concat", "window"):
            raise ValueError(f"bad head spec {entry!r}: use name:head[:key=value,...] with head = late | concat | window")
        kw = {}
        for item in (parts[2].split(",") if len(parts) > 2 and parts[2] else []):
            key, _, val = item.partition("=")
            if key not in _CASTS or not val:
                raise ValueError(f"bad option {item!r} in {entry!r}; options: {sorted(_CASTS)}")
            kw[key] = _CASTS[key](val)
        if kw.get("lidar_input", "feat") not in ("feat", "prob") or kw.get("image_input", "feat") not in ("feat", "prob") \
                or (kw.get("image_input", "feat") == "prob" and parts[1] == "window") or len(parts) > 3:
            raise ValueError(f"bad head spec {entry!r}: lidar_input / image_input are feat | prob (image_input=prob only "
                             "for concat), options are separated by ','")
        specs.append(HeadSpec(name=parts[0], head=parts[1], **kw))
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("every head spec needs its own name")
    return specs


# ----------------------------------------------------------------------------- one zone on the device
class ZoneBank:
    """The stored subset of one zone on the device: labels, weights, frozen LiDAR feature / logits, clean rows."""

    def __init__(self, sub: dict, xyz: Optional[np.ndarray] = None, device: str = "cuda"):
        self.zone = str(sub["zone"]); self.n = len(sub["idx"]); self.device = device
        self.labels = torch.from_numpy(sub["labels"].astype(np.int64)).to(device)
        self.w = torch.from_numpy(sub["w"].astype(np.float32)).to(device)
        self.feat = torch.from_numpy(sub["feat"]).to(device)                 # (M, 72) float16
        self.logits = torch.from_numpy(sub["logits"]).to(device)             # (M, 11) float16
        self.rows = subset_rows(sub, device)
        self.xyz = None if xyz is None else torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)).to(device)
        self.cw = torch.cumsum(self.w.double(), 0)                           # for natural-frequency sampling

    def sample(self, n: int, gen: torch.Generator, natural_frac: float = 1.0) -> torch.Tensor:
        """n point indices: with probability natural_frac proportional to the inverse sampling rate, else uniform."""
        r = torch.rand(n, 3, device=self.device, generator=gen, dtype=torch.float64)
        nat = torch.searchsorted(self.cw, r[:, 0] * self.cw[-1]).clamp_(max=self.n - 1)
        uni = (r[:, 1] * self.n).long().clamp_(max=self.n - 1)
        return torch.where(r[:, 2] < natural_frac, nat, uni)

    def take_rows(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.rows.items()}


@dataclass
class BankContext:
    zone: str
    maps: ZoneMaps
    bank: ZoneBank
    cal: object
    S: Optional[torch.Tensor] = None
    R: Optional[torch.Tensor] = None

    def free(self):
        self.maps = None; self.bank = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_bank_context(cache_dir: str, cache2d_dir: str, data_dir: str, subset_dir: str, zone: str,
                      need_xyz: bool = False, with_feat: bool = True, device: str = "cuda",
                      chunk: int = 20_000_000, verbose: bool = True) -> BankContext:
    """Maps + bank of a zone. The image order is the one stored in the bank (its rows index into it). The camera
    poses and the coordinates of the bank points (from the LiDAR cache) are loaded only for pose-noise augmentation,
    which re-projects the stored views and keeps their visibility - no depth maps."""
    t0 = time.time()
    path = os.path.join(subset_dir, zone, "subset.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no feature bank for {zone}: {path} (build it with train_heads_bank.py --build-only)")
    sub = load_subset(path)
    image_ids = [str(x) for x in sub["image_ids"]]
    cal, poses = load_zone_cameras(os.path.join(data_dir, zone))
    maps = ZoneMaps(cache2d_dir, zone, image_ids, device=device, verbose=False, with_feat=with_feat)
    S = R = xyz_sub = None
    if need_xyz:
        with np.load(os.path.join(cache_dir, zone, "points.npz")) as p:
            xyz_sub = p["xyz"][sub["idx"]].astype(np.float32); origin = p["origin"]
        S, R = zone_cams(poses, image_ids, origin, device)
    bank = ZoneBank(sub, xyz_sub, device)
    if verbose:
        print(f"[bank] {zone}: {bank.n:,} bank points, {len(image_ids)} images"
              + (", cameras + coordinates" if need_xyz else "") + f"  (loaded in {(time.time() - t0) / 60:.1f} min)", flush=True)
    return BankContext(zone=zone, maps=maps, bank=bank, cal=cal, S=S, R=R)


# ----------------------------------------------------------------------------- metrics on weighted confusions
def cm_metrics(cm: np.ndarray) -> dict:
    inter = np.diag(cm); union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), np.nan)
    present = cm.sum(1) > 0
    return {"mIoU": float(100 * np.nanmean(iou)) if np.isfinite(iou).any() else float("nan"),
            "mIoU_present": float(100 * np.nanmean(iou[present])) if present.any() else float("nan"),
            "OA": float(100 * inter.sum() / max(cm.sum(), 1e-12)),
            "per_class": {n: (None if np.isnan(v) else float(100 * v)) for n, v in zip(CLASS_NAMES, iou)}}


def result_metrics(d: dict) -> dict:
    """Metrics of one model from its pooled + per-zone confusions ({"cm", "zones"}): mIoU, OA, per-class IoU and the
    per-zone mIoU over the classes present in each zone."""
    m = cm_metrics(d["cm"])
    return {"mIoU": m["mIoU"], "OA": m["OA"], "per_class": m["per_class"],
            "per_zone": {z: cm_metrics(cm)["mIoU_present"] for z, cm in d["zones"].items()}}


# ----------------------------------------------------------------------------- one head = one experiment = one name
def run_cfg(spec: HeadSpec, base_cfg: dict) -> dict:
    """The configuration stored with a head: the settings of the group it is trained in + its own spec."""
    return dict(base_cfg, **{"head": spec.head, "k": spec.k, "lidar_input": spec.lidar_input, "image_input": spec.image_input,
                             "uniform_views": spec.uniform_views, "lidar_drop": spec.lidar_drop,
                             "pose_noise_t": spec.pose_t, "pose_noise_r": spec.pose_r, "aug_prob": spec.aug_prob,
                             "natural_frac": spec.natural_frac, "seed": spec.seed, "lr": spec.lr, "wd": spec.wd,
                             "trainer": "bank", "name": spec.name})


def identity_conflict(work_dir: str, cfg: dict) -> Optional[str]:
    """None when <work_dir>/last.pth is absent or was written by the bank trainer for the same experiment
    (`_IDENTITY` settings); otherwise the reason why the folder must not be reused."""
    path = os.path.join(work_dir, "last.pth")
    if not os.path.exists(path):
        return None
    stored = torch.load(path, map_location="cpu", weights_only=False)["cfg"]
    if stored.get("trainer") != "bank":
        return "it holds a head that was not trained by the bank trainer"
    diff = {k: (stored.get(k), cfg.get(k)) for k in _IDENTITY if stored.get(k) != cfg.get(k)}
    return f"it holds a head trained with another configuration (stored -> requested: {diff})" if diff else None


def check_names(specs: Sequence[HeadSpec], base_cfg: dict, out_root: str) -> None:
    """Raise - before any expensive step - when a head name is already taken by another experiment."""
    for sp in specs:
        work = os.path.join(out_root, sp.name)
        why = identity_conflict(work, run_cfg(sp, base_cfg))
        if why:
            raise ValueError(f"{work}: {why}. Give the new head another name, or delete that folder to train it again.")


# ----------------------------------------------------------------------------- trainer
class _Run:
    def __init__(self, spec: HeadSpec, total_iters: int, base_cfg: dict, out_root: str, device: str):
        self.spec, self.device = spec, device
        self.work_dir = os.path.join(out_root, spec.name)
        torch.manual_seed(spec.seed)
        self.head = spec.build().to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=spec.lr, weight_decay=spec.wd)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, max_lr=spec.lr, total_steps=max(total_iters, 1),
                                                         pct_start=base_cfg.get("warmup", 0.05), div_factor=10, final_div_factor=100)
        self.gen = torch.Generator(device=device); self.gen.manual_seed(spec.seed + 11)      # LiDAR dropout
        self.it, self.best, self.passes_done, self.loss = 0, -1.0, 0, float("nan")
        self.best_state = None
        self.cfg = run_cfg(spec, base_cfg)

    def state(self, weights: Optional[dict] = None) -> dict:
        return {"head": weights if weights is not None else self.head.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "it": self.it, "best": self.best, "passes_done": self.passes_done,
                "cfg": self.cfg, "gen": self.gen.get_state().cpu()}

    def save(self, name: str, weights: Optional[dict] = None):
        os.makedirs(self.work_dir, exist_ok=True)
        path = os.path.join(self.work_dir, name)
        torch.save(self.state(weights), path + ".tmp"); os.replace(path + ".tmp", path)
        cfg_path = os.path.join(self.work_dir, "config.json")
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w") as fh:
                json.dump(self.cfg, fh, indent=1)

    def try_resume(self) -> bool:
        path = os.path.join(self.work_dir, "last.pth")
        if not os.path.exists(path):
            return False
        why = identity_conflict(self.work_dir, self.cfg)
        if why:
            raise ValueError(f"{self.work_dir}: {why}. Give the new head another name, or delete that folder to train it again.")
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.head.load_state_dict(st["head"]); self.opt.load_state_dict(st["opt"]); self.sched.load_state_dict(st["sched"])
        self.it, self.best, self.passes_done = st["it"], st["best"], st["passes_done"]
        self.gen.set_state(st["gen"].cpu())                       # a generator state is always a CPU ByteTensor
        best = os.path.join(self.work_dir, "best.pth")
        if os.path.exists(best):
            self.best_state = torch.load(best, map_location=self.device, weights_only=False)["head"]
        return True


class _Ref:
    """An existing checkpoint evaluated next to the trained heads: never trained, nothing is written to its folder."""

    def __init__(self, name: str, head: nn.Module):
        self.spec = HeadSpec(name=name, head=type(head).__name__); self.head = head


class BankTrainer:
    """Trains every head of `specs` on the banks of cfg['train_zones'] (zone-major) and validates on the banks of
    cfg['val_zones'] after each pass. cfg keys: train_zones, val_zones, passes, its_per_zone, batch_points, chunk,
    warmup, clip, lovasz, tau, seed_order (zone order and augmentation decisions; 'seed' is read when it is absent)."""

    def __init__(self, specs: Sequence[HeadSpec], class_w: Optional[np.ndarray], cfg: dict, out_root: str, device: str = "cuda"):
        self.cfg, self.device, self.out_root = cfg, device, out_root
        self.total_iters = cfg["passes"] * len(cfg["train_zones"]) * cfg["its_per_zone"]
        self.crit = SegLoss(class_w, ce_weight=1.0, lovasz_weight=cfg.get("lovasz", 1.0)).to(device)
        self.runs = [_Run(s, self.total_iters, cfg, out_root, device) for s in specs]
        self.refs: List[_Ref] = []
        self.order_seed = int(cfg.get("seed_order", cfg.get("seed", 0)))
        self.rng = np.random.default_rng(self.order_seed + 7)                       # augmentation decisions
        self.aug_gen = torch.Generator(device=device); self.aug_gen.manual_seed(self.order_seed + 17)

    def add_references(self, heads: Dict[str, nn.Module]) -> None:
        """Existing heads (name -> module, e.g. from ProtocolModels.from_checkpoints) that every evaluation of all
        heads also reports: the bank evaluation of crop-trained heads, or the best heads of an earlier group."""
        taken = {r.spec.name for r in self.runs + self.refs} | {"lidar", "imagevote"}
        for name, head in heads.items():
            if name in taken:
                raise ValueError(f"reference head {name!r} has the name of a head that is being trained or of a baseline")
            self.refs.append(_Ref(name, head.to(self.device).eval())); taken.add(name)

    # ------------------------------------------------------------------ bookkeeping
    @property
    def need_xyz(self) -> bool:
        return any(r.spec.pose_t > 0 or r.spec.pose_r > 0 for r in self.active)

    @property
    def with_feat(self) -> bool:
        return any(getattr(r.head, "uses_feat", True) for r in self.runs + self.refs)

    @property
    def active(self) -> List[_Run]:
        return [r for r in self.runs if r.passes_done < self.cfg["passes"]]

    def resume(self) -> int:
        """Load finished / partly trained heads. Returns the pass to start from: unfinished heads must share it,
        otherwise they start over together."""
        for r in self.runs:
            if r.try_resume():
                print(f"[bank] {r.spec.name}: resumed at it {r.it} ({r.passes_done}/{self.cfg['passes']} passes, best {r.best:.2f})", flush=True)
        todo = self.active
        done = {r.passes_done for r in todo}
        if len(done) > 1:
            print("[bank] unfinished heads are at different passes -> restarting them together", flush=True)
            fresh = {r.spec.name: _Run(r.spec, self.total_iters, self.cfg, self.out_root, self.device) for r in todo}
            self.runs = [fresh.get(r.spec.name, r) for r in self.runs]
            return 0
        return done.pop() if done else self.cfg["passes"]

    def zone_order(self, pass_i: int) -> List[str]:
        names = list(self.cfg["train_zones"])
        perm = np.random.default_rng(self.order_seed + 1000 + pass_i).permutation(len(names))
        return [names[i] for i in perm]

    def _sample_gen(self, seed: int, pass_i: int, zone: str) -> torch.Generator:
        g = torch.Generator(device=self.device)
        g.manual_seed(1_000_003 * seed + 10_007 * pass_i + sum(ord(c) for c in zone))
        return g

    # ------------------------------------------------------------------ training on one zone
    def train_zone(self, ctx: BankContext, its: int, pass_i: int = 0) -> None:
        cfg, bank = self.cfg, ctx.bank
        runs = self.active
        groups: Dict[tuple, List[_Run]] = {}
        for r in runs:
            r.head.train(); groups.setdefault(r.spec.group(), []).append(r)
        gens = {key: self._sample_gen(key[0], pass_i, ctx.zone) for key in groups}       # same seed -> same batches
        B, chunk = cfg.get("batch_points", 120_000), cfg.get("chunk", 60_000)
        for _ in range(its):
            for key, members in groups.items():
                seed, pose_t, pose_r, aug_prob, natural_frac = key
                idx = bank.sample(B, gens[key], natural_frac)
                rows = bank.take_rows(idx)
                if (pose_t > 0 or pose_r > 0) and self.rng.random() < aug_prob:
                    st, sr = float(self.rng.uniform(0, pose_t)), float(self.rng.uniform(0, pose_r))
                    S2, R2 = perturb_cams(ctx.S, ctx.R, st, sr, self.aug_gen)
                    rows = reproject_rows(rows, bank.xyz[idx], S2, R2, ctx.cal)     # visibility kept (protocol mode "fast")
                labels = bank.labels[idx]
                kmax = max([r.head.window for r in members if getattr(r.head, "uses_feat", True)] + [1])
                for r in members:
                    r.opt.zero_grad(set_to_none=True); r.loss = 0.0
                n_chunks = (B + chunk - 1) // chunk
                for s in range(0, B, chunk):
                    sel = slice(s, min(B, s + chunk))
                    rc = {k: v[sel] for k, v in rows.items()}
                    with torch.no_grad():
                        img_prob, img_feat = lift(ctx.maps, rc, window=kmax)
                        lg0, ft0 = bank.logits[idx[sel]].float(), bank.feat[idx[sel]].float()
                    for r in members:
                        lg, ft, _ = drop_lidar(lg0, ft0, r.spec.lidar_drop, r.gen)
                        f_in = window_slice(img_feat, kmax, r.head.window) if getattr(r.head, "uses_feat", True) else None
                        with torch.autocast("cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
                            out = r.head(lg, ft, img_prob, f_in, rc["desc"], rc["mask"])
                        loss = self.crit(out.float(), labels[sel]) / n_chunks
                        if torch.isfinite(loss):
                            loss.backward(); r.loss += float(loss.detach())
                        else:
                            print(f"[bank] {r.spec.name}: non-finite loss at it {r.it}; chunk skipped", flush=True)
                for r in members:
                    torch.nn.utils.clip_grad_norm_(r.head.parameters(), cfg.get("clip", 1.0))
                    r.opt.step(); r.sched.step(); r.it += 1

    # ------------------------------------------------------------------ prediction / validation
    @torch.no_grad()
    def predict(self, ctx: BankContext, runs: Sequence[_Run], ablate: Optional[str] = None, chunk: int = 100_000,
                references: bool = True) -> Dict[str, torch.Tensor]:
        """Predicted class of every bank point per head (+ 'lidar' and 'imagevote' references).
        ablate: None | 'no_lidar' (the head gets no LiDAR evidence) | 'no_image' (no view is visible)."""
        bank, M = ctx.bank, ctx.bank.n
        names = [r.spec.name for r in runs] + (["lidar", "imagevote"] if references else [])
        preds = {n: torch.empty(M, dtype=torch.uint8, device=self.device) for n in names}
        for r in runs:
            r.head.eval()
        kmax = max([r.head.window for r in runs if getattr(r.head, "uses_feat", True)] + [1])
        for s in range(0, M, chunk):
            sel = slice(s, min(M, s + chunk))
            rc = {k: v[sel] for k, v in bank.rows.items()}
            img_prob, img_feat = lift(ctx.maps, rc, window=kmax)
            lg, ft = bank.logits[sel].float(), bank.feat[sel].float()
            if references:
                preds["lidar"][sel] = lg.argmax(1).to(torch.uint8)
                p = (img_prob * uniform_weights(rc["mask"])[..., None]).sum(1)
                pv = p.argmax(1).to(torch.uint8); pv[~rc["mask"].any(1)] = IMAGEVOTE_FALLBACK
                preds["imagevote"][sel] = pv
            mask = rc["mask"]
            if ablate == "no_lidar":
                lg, ft, _ = drop_lidar(lg, ft, 1.0)
            elif ablate == "no_image":
                mask = torch.zeros_like(mask); img_prob = torch.zeros_like(img_prob)
                img_feat = None if img_feat is None else torch.zeros_like(img_feat)
            for r in runs:
                f_in = window_slice(img_feat, kmax, r.head.window) if getattr(r.head, "uses_feat", True) else None
                with torch.autocast("cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
                    out = r.head(lg, ft, img_prob, f_in, rc["desc"], mask)
                preds[r.spec.name][sel] = out.float().argmax(1).to(torch.uint8)
        return preds

    def evaluate_cases(self, loader, zones: Sequence[str], cases: Sequence[Optional[str]] = (None,),
                       runs: Optional[Sequence[_Run]] = None) -> Dict[str, Dict[str, dict]]:
        """One pass over the zones (each loaded once), one prediction per case (None = full input, 'no_lidar',
        'no_image'). Returns {case or 'full': {name: {"cm": pooled confusion, "zones": {zone: confusion}}}}.
        runs = None: every trained head and every reference head."""
        runs = list(self.runs + self.refs if runs is None else runs)
        out: Dict[str, Dict[str, dict]] = {(c or "full"): {} for c in cases}
        for z in zones:
            ctx = loader(z, False, self.with_feat)
            for c in cases:
                preds = self.predict(ctx, runs, ablate=c)
                for name, p in preds.items():
                    cm = weighted_confusion(ctx.bank.labels, p, ctx.bank.w)
                    d = out[c or "full"].setdefault(name, {"cm": np.zeros((11, 11)), "zones": {}})
                    d["cm"] += cm; d["zones"][z] = cm
            ctx.free()
        return out

    def evaluate(self, loader, zones: Sequence[str], runs: Optional[Sequence[_Run]] = None, verbose: bool = True) -> Dict[str, dict]:
        """{name: {"cm": pooled confusion, "zones": {zone: confusion}}} for every head and the two references."""
        out = self.evaluate_cases(loader, zones, (None,), runs)["full"]
        if verbose:
            for name, d in out.items():
                m = cm_metrics(d["cm"])
                print(f"[bank-eval] {name:18s} mIoU {m['mIoU']:.2f}  OA {m['OA']:.2f}", flush=True)
        return out

    def summary(self, run: _Run, res: Dict[str, dict], tag: str) -> dict:
        """eval.json content for one head (same schema as FusionTrainer.evaluate + the ImageVote reference)."""
        d, lid, iv = res[run.spec.name], res["lidar"], res["imagevote"]
        m = cm_metrics(d["cm"])
        per_zone = {}
        for z, cm in d["zones"].items():
            mz = cm_metrics(cm)
            per_zone[z] = {"mIoU": mz["mIoU_present"], "OA": mz["OA"], "per_class": mz["per_class"],
                           "lidar_only_mIoU": cm_metrics(lid["zones"][z])["mIoU_present"],
                           "imagevote_mIoU": cm_metrics(iv["zones"][z])["mIoU_present"]}
        ml, mi = cm_metrics(lid["cm"]), cm_metrics(iv["cm"])
        return {"model": run.spec.head, "name": run.spec.name, "tag": tag, "it": run.it, "mIoU": m["mIoU"], "OA": m["OA"],
                "per_class": m["per_class"], "per_zone": per_zone, "confusion": d["cm"].tolist(), "class_names": CLASS_NAMES,
                "lidar_only": {"mIoU": ml["mIoU"], "OA": ml["OA"], "per_class": ml["per_class"]},
                "imagevote": {"mIoU": mi["mIoU"], "OA": mi["OA"], "per_class": mi["per_class"]},
                "estimator": "feature bank: class-stratified subset, inverse-sampling-rate weights", "spec": asdict(run.spec)}

    # ------------------------------------------------------------------ schedule
    def fit(self, loader, max_hours: float = 1e9) -> str:
        """loader(zone, need_xyz, with_feat) -> BankContext. Returns 'done' or 'time'."""
        cfg, t0 = self.cfg, time.time()
        start = self.resume()
        if self.active:
            print(f"[bank] training {len(self.active)} heads in {len({r.spec.group() for r in self.active})} batch groups: passes "
                  f"{start + 1}..{cfg['passes']} x {len(cfg['train_zones'])} zones x {cfg['its_per_zone']} its of "
                  f"{cfg.get('batch_points', 120_000):,} points", flush=True)
        for pass_i in range(start, cfg["passes"]):
            for zone in self.zone_order(pass_i):
                t1 = time.time()
                ctx = loader(zone, self.need_xyz, self.with_feat)
                t2 = time.time()
                self.train_zone(ctx, cfg["its_per_zone"], pass_i)
                ctx.free()
                act = self.active
                print(f"[bank] pass {pass_i + 1}/{cfg['passes']} {zone}: load {(t2 - t1) / 60:.1f} min, {cfg['its_per_zone']} its x "
                      f"{len(act)} heads in {(time.time() - t2) / 60:.1f} min ({(time.time() - t2) / max(cfg['its_per_zone'], 1):.2f} s/it)"
                      f"   loss " + "  ".join(f"{r.spec.name} {r.loss:.3f}" for r in act)
                      + f"   (total {(time.time() - t0) / 60:.1f} min)", flush=True)
            trained = self.active
            res = self.evaluate(loader, cfg["val_zones"], runs=trained)
            for r in trained:
                r.passes_done = pass_i + 1
                s = self.summary(r, res, tag="best")
                if s["mIoU"] > r.best:
                    r.best = s["mIoU"]; r.best_state = copy.deepcopy(r.head.state_dict())
                    r.save("best.pth")
                    os.makedirs(os.path.join(r.work_dir, "eval_bank"), exist_ok=True)
                    with open(os.path.join(r.work_dir, "eval_bank", "eval.json"), "w") as fh:
                        json.dump(s, fh, indent=1)
                r.save("last.pth")
            if (time.time() - t0) / 3600 > max_hours and pass_i + 1 < cfg["passes"]:
                return "time"
        return "done"

    def load_best(self) -> None:
        for r in self.runs:
            if r.best_state is not None:
                r.head.load_state_dict(r.best_state)

    def diagnostics(self, loader, train_zones: Sequence[str] = ()) -> dict:
        """Best weights of every head on the validation banks with full input, without LiDAR evidence and without
        image evidence, and (optionally) with full input on training-zone banks. Returns
        {case: {name: {"mIoU", "OA", "per_class", "per_zone"}}} with cases val, val_no_lidar, val_no_image[, train]:
        which branch a head relies on for which class, and how much better both branches are on the zones the
        backbones were trained on. Reference heads are included. Written per trained head to
        <work_dir>/eval_bank/diagnostics.json."""
        self.load_best()
        res = self.evaluate_cases(loader, self.cfg["val_zones"], (None, "no_lidar", "no_image"))
        cases = {"val": res["full"], "val_no_lidar": res["no_lidar"], "val_no_image": res["no_image"]}
        if train_zones:
            cases["train"] = self.evaluate_cases(loader, list(train_zones))["full"]
        out = {case: {name: result_metrics(d) for name, d in r.items()} for case, r in cases.items()}
        for r in self.runs:
            os.makedirs(os.path.join(r.work_dir, "eval_bank"), exist_ok=True)
            mine = {case: {n: m for n, m in v.items() if n in (r.spec.name, "lidar", "imagevote")} for case, v in out.items()}
            with open(os.path.join(r.work_dir, "eval_bank", "diagnostics.json"), "w") as fh:
                json.dump(mine, fh, indent=1)
        return out


def bank_class_counts(subset_dir: str, zones: Sequence[str]) -> np.ndarray:
    """Class totals of the zones from their banks: the inverse-sampling-rate weights sum to the full-zone counts,
    so no point cloud has to be read."""
    cnt = np.zeros(11, np.float64)
    for z in zones:
        with np.load(os.path.join(subset_dir, z, "subset.npz"), allow_pickle=False) as d:       # only two small arrays are read
            labels, w = d["labels"], d["w"]
        cnt += np.bincount(labels.astype(np.int64), weights=w.astype(np.float64), minlength=11)[:11]
    return cnt


def zone_class_counts(subset_dir: str, zones: Sequence[str]) -> Dict[str, np.ndarray]:
    """Estimated number of points per class in every zone, from the bank weights."""
    return {z: bank_class_counts(subset_dir, [z]) for z in zones}


def format_counts(counts: Dict[str, np.ndarray]) -> str:
    """Plain-text table: thousands of points per class (columns) and zone (rows); '.' = class absent."""
    head = f"{'zone':8s} " + " ".join(f"{c[:9]:>9s}" for c in CLASS_NAMES)
    lines = [head, "-" * len(head)]
    for z, c in counts.items():
        lines.append(f"{z:8s} " + " ".join(f"{'.' if v <= 0 else format(v / 1e3, '.1f' if v < 1e4 else '.0f'):>9s}" for v in c))
    return "\n".join(lines)


def format_table(metrics: Dict[str, dict], order: Optional[Sequence[str]] = None) -> str:
    """Plain-text table: one row per model with mIoU, OA and the per-class IoUs."""
    names = [n for n in (order or metrics) if n in metrics]
    head = f"{'model':20s} {'mIoU':>6s} {'OA':>6s}  " + " ".join(f"{c[:9]:>9s}" for c in CLASS_NAMES)
    lines = [head, "-" * len(head)]
    for n in names:
        m = metrics[n]
        cells = " ".join(f"{'--' if m['per_class'][c] is None else format(m['per_class'][c], '.1f'):>9s}" for c in CLASS_NAMES)
        lines.append(f"{n:20s} {m['mIoU']:6.2f} {m['OA']:6.2f}  {cells}")
    return "\n".join(lines)
