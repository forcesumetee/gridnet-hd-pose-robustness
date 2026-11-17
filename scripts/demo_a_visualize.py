"""
Demo A (step 1) — GridNet-HD zone viewer for a laptop (CPU only)

What it does
  1. reads one LAS zone with laspy
  2. prints per-class point counts and percentages (11 training classes + unlabeled)
  3. writes a class-colored PLY you can open in CloudCompare
  4. optionally opens an Open3D window (downsampled so a MacBook can rotate it smoothly)

Usage
  python demo_a_visualize.py --las path/to/t5b5.las
  python demo_a_visualize.py --las path/to/t5b5.las --voxel 0.2 --no-window

Install
  pip install "laspy[lazrs]" numpy open3d
"""

import argparse
import sys

import numpy as np

try:
    import laspy
except ImportError:
    sys.exit("laspy missing:  pip install 'laspy[lazrs]'")

# 11 training classes of GridNet-HD (+255 unlabeled), colors chosen to be distinguishable
CLASSES = {
    0: ("Pylon", (0.10, 0.10, 0.10)),
    1: ("Conductor cable", (1.00, 0.55, 0.00)),
    2: ("Structural cable", (0.90, 0.10, 0.10)),
    3: ("Insulator", (0.80, 0.00, 0.80)),
    4: ("High vegetation", (0.00, 0.45, 0.00)),
    5: ("Low vegetation", (0.30, 0.75, 0.30)),
    6: ("Herbaceous vegetation", (0.75, 0.90, 0.55)),
    7: ("Rock, gravel, soil", (0.55, 0.45, 0.35)),
    8: ("Impervious soil (road)", (0.50, 0.50, 0.50)),
    9: ("Water", (0.10, 0.40, 0.95)),
    10: ("Building", (0.95, 0.85, 0.20)),
    255: ("Unlabeled", (0.85, 0.85, 0.85)),
}

# The LAS field "ground_truth" stores the ORIGINAL 22 classes (paper, supplementary Table 4).
ORIGINAL = {
    0: "Pylon foundation", 1: "Cat head type pylon", 2: "Triangle-arm pylon", 3: "Portal pylon",
    4: "Other pylon", 5: "Conductor cable", 6: "Guard cable", 7: "Anchor cable",
    8: "Suspension insulator - glass", 9: "Strain insulator - glass",
    10: "Suspension insulator - porcelain", 11: "Strain insulator - porcelain",
    12: "Other insulator", 13: "Signage", 14: "High vegetation", 15: "Low vegetation",
    16: "Herbaceous vegetation", 17: "Rock", 18: "Gravel, soil", 19: "Impervious soil (Road)",
    20: "Water", 21: "Building", 255: "Unlabeled",
}
REMAP = np.full(256, 255, dtype=np.uint8)
for orig, train in {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3,
                    14: 4, 15: 5, 16: 6, 17: 7, 18: 7, 19: 8, 20: 9, 21: 10}.items():
    REMAP[orig] = train


def is_original_space(labels):
    """Original IDs use 11..21 for vegetation/ground/building; the 11-class space stops at 10."""
    u = np.unique(labels)
    return bool(((u >= 11) & (u <= 21)).any())


def to_training_classes(labels):
    return REMAP[labels] if is_original_space(labels) else labels


def read_zone(path, label_field=None):
    """Return xyz (N,3) float32 and labels (N,) uint8. Auto-detects the label field."""
    las = laspy.read(path)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    # keep coordinates small (UTM values are ~1e5–1e6; float32 would lose precision)
    xyz -= xyz.min(axis=0)
    xyz = xyz.astype(np.float32)

    candidates = [label_field] if label_field else []
    candidates += ["ground_truth", "classification"]
    extra = list(las.point_format.extra_dimension_names)
    labels = None
    for name in candidates:
        if name is None:
            continue
        if name in extra or name == "classification":
            labels = np.asarray(las[name]).astype(np.int32)
            print(f"[info] label field used: '{name}'")
            break
    if labels is None:
        sys.exit(f"no label field found; extra dims are: {extra}")
    return xyz, labels


def print_stats(labels, names, title):
    total = labels.size
    print(f"\n{title}\n{'ID':>4}  {'class':<34}{'points':>14}{'%':>9}")
    for cid in sorted(set(labels.tolist())):
        n = int((labels == cid).sum())
        name = names.get(cid, "?")
        print(f"{cid:>4}  {name:<34}{n:>14,}{100.0 * n / total:>8.3f}%")
    print(f"{'':>4}  {'TOTAL':<34}{total:>14,}\n")


def colorize(labels):
    colors = np.full((labels.size, 3), 0.85, dtype=np.float32)
    for cid, (_, rgb) in CLASSES.items():
        colors[labels == cid] = rgb
    return colors


def write_ply(path, xyz, rgb, labels):
    """Binary PLY with xyz, rgb and a 'label' scalar field - no open3d needed."""
    n = len(labels)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("label", "u1")])
    rec = np.empty(n, dtype=dt)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rec["label"] = labels.astype(np.uint8)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property uchar label\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        rec.tofile(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--las", required=True, help="path to one zone .las/.laz")
    ap.add_argument("--label-field", default=None, help="override label field name")
    ap.add_argument("--voxel", type=float, default=0.10,
                    help="voxel size (m) for the viewer only; 0 = no downsampling")
    ap.add_argument("--ply", default=None, help="output PLY path (default: <las>_classes.ply)")
    ap.add_argument("--no-window", action="store_true", help="skip the Open3D window")
    args = ap.parse_args()

    xyz, raw = read_zone(args.las, args.label_field)
    if is_original_space(raw):
        print_stats(raw, ORIGINAL, "Original 22-class labels (as stored in the LAS)")
        labels = REMAP[raw]
        print_stats(labels, {k: v[0] for k, v in CLASSES.items()},
                    "Remapped to the 11 training classes (paper Table 4)")
    else:
        labels = raw
        print_stats(labels, {k: v[0] for k, v in CLASSES.items()}, "11 training classes")

    colors = colorize(labels)
    out = args.ply or args.las.rsplit(".", 1)[0] + "_classes.ply"
    write_ply(out, xyz, (colors * 255).astype(np.uint8), labels)
    print(f"[info] wrote {out}  (open it in CloudCompare; scalar field 'label' = 11-class id)")

    if args.no_window:
        return
    try:
        import open3d as o3d
    except Exception as e:  # report the real reason, not just "missing"
        print(f"[warn] open3d could not be imported ({type(e).__name__}: {e}); "
              f"skipping the window - open the PLY in CloudCompare instead")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    view = pcd.voxel_down_sample(args.voxel) if args.voxel > 0 else pcd
    print(f"[info] viewer points: {len(view.points):,}  (voxel {args.voxel} m)")
    o3d.visualization.draw_geometries([view], window_name="GridNet-HD zone by class",
                                      width=1400, height=900)


if __name__ == "__main__":
    main()
