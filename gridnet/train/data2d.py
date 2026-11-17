"""Image/mask data for the 2D branch: zone listing, crop sampling with rare-class centring, class weights,
and a fetch helper for the Hugging Face dataset.

Masks on disk are the authors' projected 3D labels (original 22-class ids); `gridnet.io.load_mask` remaps them
to the 11 training classes with 255 = unlabelled (pixels without a LiDAR point), which the loss ignores.
"""
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from ..classes import NUM_CLASSES, REMAP
from ..io import image_path, list_image_ids

Image.MAX_IMAGE_PIXELS = None
RARE = (0, 1, 2, 3)                 # pylon, conductor, structural cable, insulator
CENTRE_CLASSES = (0, 1, 2, 3, 8, 9, 10)   # classes a crop may be centred on (electrical + road, water, building)


def load_mask(zone_dir: str, image_id: str) -> np.ndarray:
    """Authors' mask (original 22-class ids) remapped to the 11 classes, 255 = unlabelled.
    Direct REMAP lookup (no np.unique scan): ~20x faster than gridnet.io.load_mask on a 21 MP mask."""
    m = np.asarray(Image.open(os.path.join(zone_dir, "masks", image_id + ".png")))
    if m.ndim == 3:
        m = m[..., 0]
    return REMAP[m.astype(np.uint8)]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def list_pairs(root: str, zones: Sequence[str], per_zone: Optional[int] = None, seed: int = 0) -> List[Tuple[str, str]]:
    """[(zone, image_id)] for every image that has a mask; optionally a fixed random subset per zone."""
    rng = random.Random(seed)
    pairs = []
    for z in zones:
        zd = os.path.join(root, z)
        if not os.path.isdir(os.path.join(zd, "images")):
            continue
        ids = [i for i in list_image_ids(zd) if os.path.exists(os.path.join(zd, "masks", i + ".png"))]
        if per_zone and len(ids) > per_zone:
            ids = sorted(rng.sample(ids, per_zone))
        pairs += [(z, i) for i in ids]
    return pairs


def class_weights_2d(root: str, pairs: Sequence[Tuple[str, str]], n: int = 100, power: float = 0.5,
                     seed: int = 0, stride: int = 8) -> np.ndarray:
    """(freq ** -power) normalised to mean 1, from a subsample of masks (every `stride`-th pixel)."""
    rng = random.Random(seed)
    sample = pairs if len(pairs) <= n else rng.sample(list(pairs), n)
    counts = np.zeros(NUM_CLASSES, np.int64)
    for z, i in sample:
        m = load_mask(os.path.join(root, z), i)[::stride, ::stride]
        counts += np.bincount(m[m != 255].astype(np.int64), minlength=NUM_CLASSES)
    freq = counts / max(counts.sum(), 1)
    w = np.where(freq > 0, np.power(np.maximum(freq, 1e-8), -power), 0.0)
    w = np.where(freq > 0, w / w[freq > 0].mean(), 1.0)
    return w.astype(np.float32)


def _open_scaled(path: str, scale: float) -> Image.Image:
    img = Image.open(path)
    if scale < 1.0:
        w, h = img.size
        img.draft("RGB", (int(w * scale), int(h * scale)))       # fast JPEG DCT downscale
        img = img.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    else:
        img = img.convert("RGB")
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class CropDataset(torch.utils.data.Dataset):
    """Random (train) or fixed (val) crops. Half of the training crops are centred on a rare-class pixel."""

    def __init__(self, root: str, pairs: Sequence[Tuple[str, str]], crop: int = 1024, train: bool = True,
                 scales: Sequence[float] = (0.5, 0.75, 1.0), rare_prob: float = 0.5, seed: int = 0,
                 crops_per_image: int = 1):
        self.root, self.pairs, self.crop, self.train = root, list(pairs), crop, train
        self.scales, self.rare_prob, self.seed, self.cpi = tuple(scales), rare_prob, seed, crops_per_image

    def __len__(self):
        return len(self.pairs) * self.cpi

    def __getitem__(self, idx):
        z, i = self.pairs[idx % len(self.pairs)]
        rng = random.Random(None if self.train else self.seed * 100003 + idx)
        scale = rng.choice(self.scales) if self.train else 1.0
        img = np.array(_open_scaled(image_path(os.path.join(self.root, z), i), scale))
        m = load_mask(os.path.join(self.root, z), i)
        if scale < 1.0:
            m = np.array(Image.fromarray(m).resize((img.shape[1], img.shape[0]), Image.NEAREST))
        H, W = m.shape
        c = self.crop
        if H < c or W < c:                                           # pad small (down-scaled) images
            ph, pw = max(0, c - H), max(0, c - W)
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            m = np.pad(m, ((0, ph), (0, pw)), constant_values=255)
            H, W = m.shape
        y0, x0 = rng.randint(0, H - c), rng.randint(0, W - c)
        if rng.random() < self.rare_prob:
            # class-balanced centring: pick one of the centre classes present (uniformly over classes, not
            # pixels), then a pixel of that class; keeps water / road / building in the crops as well
            sub = m[::4, ::4]
            present = [k for k in CENTRE_CLASSES if (sub == k).any()]
            if present:
                ys, xs = np.where(sub == rng.choice(present))
                k = rng.randrange(len(ys))
                cy, cx = int(ys[k]) * 4, int(xs[k]) * 4
                y0 = min(max(cy - c // 2, 0), H - c)
                x0 = min(max(cx - c // 2, 0), W - c)
        img = img[y0:y0 + c, x0:x0 + c]
        m = m[y0:y0 + c, x0:x0 + c]
        if self.train and rng.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        if self.train:                                               # light photometric jitter
            gain = np.float32(rng.uniform(0.85, 1.15))
            img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return to_tensor(img), torch.from_numpy(np.ascontiguousarray(m)).long()


def load_image_tensor(root: str, zone: str, image_id: str) -> torch.Tensor:
    """Full-resolution normalised image tensor (3, H, W) for sliding-window inference."""
    img = np.array(Image.open(image_path(os.path.join(root, zone), image_id)).convert("RGB"))
    return to_tensor(img)


def fetch_zones(zones: Sequence[str], dst: str, what: Sequence[str] = ("images", "masks"),
                repo: str = "heig-vd-geo/GridNet-HD", token: Optional[str] = None) -> None:
    """Download <zone>/<what>/* from the Hugging Face dataset into dst (skips files already present)."""
    from huggingface_hub import snapshot_download
    for z in zones:
        snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{z}/{w}/*" for w in what],
                          local_dir=dst, token=token or os.environ.get("HF_TOKEN"))
