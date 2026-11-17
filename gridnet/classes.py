"""Class definitions for GridNet-HD: original 22 classes, 11 training classes, remap, palette."""
import numpy as np

# 11 training classes (paper Table 4) + 255 unlabeled
CLASSES = {
    0: "Pylon", 1: "Conductor cable", 2: "Structural cable", 3: "Insulator",
    4: "High vegetation", 5: "Low vegetation", 6: "Herbaceous vegetation",
    7: "Rock, gravel, soil", 8: "Impervious soil (road)", 9: "Water", 10: "Building",
    255: "Unlabeled",
}
NUM_CLASSES = 11
IGNORE = 255
THIN = (1, 2, 3)  # conductor, structural cable, insulator

ORIGINAL = {
    0: "Pylon foundation", 1: "Cat head type pylon", 2: "Triangle-arm pylon", 3: "Portal pylon",
    4: "Other pylon", 5: "Conductor cable", 6: "Guard cable", 7: "Anchor cable",
    8: "Suspension insulator - glass", 9: "Strain insulator - glass",
    10: "Suspension insulator - porcelain", 11: "Strain insulator - porcelain",
    12: "Other insulator", 13: "Signage", 14: "High vegetation", 15: "Low vegetation",
    16: "Herbaceous vegetation", 17: "Rock", 18: "Gravel, soil", 19: "Impervious soil (Road)",
    20: "Water", 21: "Building", 255: "Unlabeled",
}

_ORIG_TO_TRAIN = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3,
                  14: 4, 15: 5, 16: 6, 17: 7, 18: 7, 19: 8, 20: 9, 21: 10}
REMAP = np.full(256, IGNORE, dtype=np.uint8)
for _o, _t in _ORIG_TO_TRAIN.items():
    REMAP[_o] = _t

PALETTE = np.full((256, 3), 200, dtype=np.uint8)
for _c, _rgb in {0: (25, 25, 25), 1: (255, 140, 0), 2: (230, 25, 25), 3: (204, 0, 204),
                 4: (0, 115, 0), 5: (76, 190, 76), 6: (190, 230, 140), 7: (140, 115, 90),
                 8: (128, 128, 128), 9: (25, 100, 240), 10: (240, 215, 50)}.items():
    PALETTE[_c] = _rgb


def is_original_space(labels: np.ndarray) -> bool:
    """Original IDs use 11..21 for vegetation/ground/building; the 11-class space stops at 10."""
    u = np.unique(labels)
    return bool(((u >= 11) & (u <= 21)).any())


def to_training_classes(labels: np.ndarray) -> np.ndarray:
    """Remap original 22-class IDs to the 11 training classes; no-op if already remapped."""
    labels = np.asarray(labels).astype(np.uint8, copy=False)
    return REMAP[labels] if is_original_space(labels) else labels
