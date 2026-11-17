"""Predictions for zones without labels (the leaderboard's test zones): voxel-level classes of the frozen LiDAR-only
model and of a fusion head, from the same windows and the same forward passes as the validation code."""
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .data import Zone, cover_remaining, eval_windows, make_batch
from .engine import to_device
from .engine_fusion import EVAL_SEED, FrozenBackbone, fixed_torch_seed
from .fusion import build_head


@torch.no_grad()
def predict_lidar(backbone: FrozenBackbone, zone: Zone, cfg: dict, device: str = "cuda") -> np.ndarray:
    """Vote-averaged prediction of the frozen LiDAR-only model on the evaluation windows (uint8 per voxel).
    Windows, their order and the generator state are those of `FusionTrainer.predict_zone`, whose second output this
    reproduces without the image branch (LitePT draws random serialization orders in every forward pass)."""
    radius, max_points = cfg.get("radius", 25.0), cfg.get("eval_max_points", 900_000)
    votes = np.zeros((zone.n, 11), np.float32)
    counts = np.zeros(zone.n, np.int32)

    def run(idx):
        batch = to_device(make_batch([(zone.xyz[idx], zone.feats[idx], zone.labels[idx])], cfg["grid"]), device)
        _, logits = backbone(batch)
        votes[idx] += torch.softmax(logits, 1).cpu().numpy()
        counts[idx] += 1

    with fixed_torch_seed(cfg.get("eval_seed", EVAL_SEED)):
        n_reg = 0
        for idx in eval_windows(zone, radius=radius, max_points=max_points):
            run(idx); n_reg += 1
        n_extra = 0
        for idx in cover_remaining(zone, counts, radius, max_points):
            run(idx); n_extra += 1
    if n_extra:
        print(f"[predict] {zone.name}: {n_reg} windows + {n_extra} fill-in windows", flush=True)
    return votes.argmax(1).astype(np.uint8)


def head_from_checkpoint(path: str) -> Tuple[torch.nn.Module, dict, dict]:
    """(head with its trained weights, the cfg it was trained with, {"it", "best"}) from a best.pth / last.pth of
    either head trainer; the head design (type, K, inputs) is whatever the checkpoint says."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    kw = {"uniform_views": True} if (cfg["head"] == "late" and cfg.get("uniform_views")) else {}
    head = build_head(cfg["head"], k=cfg.get("k", 3), lidar_input=cfg.get("lidar_input", "feat"),
                      image_input=cfg.get("image_input", "feat"), **kw)
    head.load_state_dict(st["head"])
    return head, cfg, {"it": st.get("it"), "best": st.get("best")}


def zone_is_labeled(labels: np.ndarray, flag: Optional[bool] = None) -> bool:
    """A zone counts as labelled when its cache says so (`labeled`, written from update35) or, for older caches,
    when any voxel carries a label."""
    return bool(flag) if flag is not None else bool((labels != 255).any())


def summarize_prediction(pred: np.ndarray, labels: np.ndarray, labeled: bool) -> Dict:
    """Class counts of a prediction and, for a labelled zone, its scores per voxel (same numbers as eval.json)."""
    from ..submission import class_counts, confusion, metrics
    out = {"counts": [int(c) for c in class_counts(pred)]}
    if labeled:
        out["metrics"] = metrics(confusion(labels, pred))
    return out
