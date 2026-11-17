"""Reading GridNet-HD zones: LAS points (chunked, cached), masks, split file."""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .classes import to_training_classes

POINT_FIELDS = ("intensity", "return_number", "number_of_returns", "scan_angle_rank",
                "scan_angle", "red", "green", "blue")


def read_split(root: str) -> Dict[str, List[str]]:
    with open(os.path.join(root, "split.json")) as fh:
        return json.load(fh)


def find_las(zone_dir: str) -> str:
    lidar_dir = os.path.join(zone_dir, "lidar")
    files = sorted(f for f in os.listdir(lidar_dir) if f.lower().endswith((".las", ".laz")))
    if not files:
        raise FileNotFoundError(f"no .las in {lidar_dir}")
    return os.path.join(lidar_dir, files[0])


def load_zone_points(zone_dir: str, use_cache: bool = True, extra: bool = True,
                     chunk: int = 5_000_000) -> Dict[str, np.ndarray]:
    """Return dict with xyz (float64, N×3), labels (uint8, 11-class), and, if extra, intensity /
    return_number / number_of_returns (as stored) when present. Cached as an .npz next to the LAS.
    Labels are remapped to the 11 training classes; the cache keeps the raw labels.
    Labels come from the LAS field `ground_truth`. The test zones do not have it: their points are all
    unlabeled (255) and `labeled` is False - the standard `classification` field is never read as a label."""
    las_path = find_las(zone_dir)
    cache = las_path.rsplit(".", 1)[0] + "_cache.npz"
    out: Optional[Dict[str, np.ndarray]] = None
    if use_cache and os.path.exists(cache):
        try:
            with np.load(cache) as d:
                out = {k: d[k] for k in d.files}
            if "labels_raw" not in out and "labels" in out:   # cache written by the demo scripts
                out["labels_raw"] = out.pop("labels")
            out.setdefault("labeled", np.array(True))         # caches written before the test zones were handled
        except Exception as e:  # truncated cache from an interrupted run
            print(f"[io] cache unreadable ({e}); rebuilding")
            os.remove(cache)
            out = None
    if out is None:
        import laspy
        with laspy.open(las_path) as reader:
            n = reader.header.point_count
            xyz = np.empty((n, 3), dtype=np.float64)
            labels = np.empty(n, dtype=np.uint8)
            extras: Dict[str, np.ndarray] = {}
            field = None
            done = 0
            for ch in reader.chunk_iterator(chunk):
                if field is None:
                    dims = set(ch.point_format.dimension_names)
                    field = "ground_truth" if "ground_truth" in dims else ""
                    if extra:
                        for f in POINT_FIELDS:
                            if f in dims:
                                extras[f] = np.empty(n, dtype=np.asarray(ch[f]).dtype)
                k = len(ch)
                xyz[done:done + k, 0] = ch.x
                xyz[done:done + k, 1] = ch.y
                xyz[done:done + k, 2] = ch.z
                labels[done:done + k] = np.asarray(ch[field]).astype(np.uint8) if field else 255
                for f, arr in extras.items():
                    arr[done:done + k] = np.asarray(ch[f])
                done += k
                print(f"\r[io] reading {os.path.basename(las_path)}: {done:,}/{n:,}", end="", flush=True)
        print()
        if not field:
            print(f"[io] {os.path.basename(las_path)} has no ground_truth field (test zone): all points unlabeled")
        out = {"xyz": xyz, "labels_raw": labels, "labeled": np.array(bool(field)), **extras}
        if use_cache:
            tmp = cache + ".tmp.npz"
            np.savez(tmp, **out)
            os.replace(tmp, cache)
    out["labels"] = to_training_classes(out["labels_raw"])
    return out


def las_point_count(las_path: str) -> int:
    import laspy
    with laspy.open(las_path) as reader:
        return int(reader.header.point_count)


def read_las_labels(las_path: str, chunk: int = 10_000_000) -> Optional[np.ndarray]:
    """The 11-class labels of every point of a LAS file in file order (uint8, 255 = unlabeled), read from the field
    `ground_truth` without loading the coordinates; None when the file has no such field (test zones)."""
    import laspy
    from .classes import REMAP
    with laspy.open(las_path) as reader:
        if "ground_truth" not in set(reader.header.point_format.dimension_names):
            return None
        out = np.empty(reader.header.point_count, dtype=np.uint8)
        done = 0
        for ch in reader.chunk_iterator(chunk):
            k = len(ch)
            out[done:done + k] = REMAP[np.asarray(ch["ground_truth"]).astype(np.uint8)]
            done += k
    return out


def load_mask(zone_dir: str, image_id: str) -> np.ndarray:
    """Authors' 2D mask remapped to the 11 training classes (uint8, H×W)."""
    m = np.array(Image.open(os.path.join(zone_dir, "masks", image_id + ".png")))
    if m.ndim == 3:
        m = m[..., 0]
    return to_training_classes(m.astype(np.uint8))


def image_path(zone_dir: str, image_id: str) -> str:
    return os.path.join(zone_dir, "images", image_id + ".JPG")


def list_image_ids(zone_dir: str) -> List[str]:
    return sorted(f[:-4] for f in os.listdir(os.path.join(zone_dir, "images")) if f.upper().endswith(".JPG"))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
