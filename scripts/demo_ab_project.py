"""
Demo A (step 2) + Demo B — GridNet-HD image-to-point projection on a laptop (CPU only)

Sub-commands
  check    verify the projection convention against the provided 2D mask of one image
           (tries the sign variants of the camera model and reports label agreement)
  overlay  project the 3D labels into one image and save a side-by-side PNG:
           [image + projected labels | provided 2D mask]
  perturb  Demo B — perturb the camera pose (rotation in degrees / translation in cm),
           measure how many cable / insulator points still land on the correct mask
           pixels, save a CSV and a plot

Examples
  python demo_ab_project.py check   --zone data/t5b5 --auto
  python demo_ab_project.py overlay --zone data/t5b5 --image DJI_20241106090100_0025_D --out overlay.png
  python demo_ab_project.py perturb --zone data/t5b5 --image DJI_20241106090100_0025_D --out perturb

Camera model: Agisoft Metashape "frame" model exactly as written in the GridNet-HD
supplementary (Sec. 9): Xc = R (M - S); x = -Xc/Zc, y = -Yc/Zc, z = -Zc; radial k1..k5,
tangential p1..p4, skew b1 b2; u = W/2 + cx + x'f + x'b1 + y'b2 ; v = H/2 + cy + y'f.
`--conv sx,sy,dz` lets you flip the signs of x, y and depth; `check` finds the right one.

Install:  pip install "laspy[lazrs]" numpy pillow matplotlib
"""

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------- classes
CLASS_NAMES = {0: "Pylon", 1: "Conductor cable", 2: "Structural cable", 3: "Insulator",
               4: "High vegetation", 5: "Low vegetation", 6: "Herbaceous vegetation",
               7: "Rock/gravel/soil", 8: "Road", 9: "Water", 10: "Building", 255: "Unlabeled"}
PALETTE = np.full((256, 3), 200, dtype=np.uint8)
for cid, rgb in {0: (25, 25, 25), 1: (255, 140, 0), 2: (230, 25, 25), 3: (204, 0, 204),
                 4: (0, 115, 0), 5: (76, 190, 76), 6: (190, 230, 140), 7: (140, 115, 90),
                 8: (128, 128, 128), 9: (25, 100, 240), 10: (240, 215, 50)}.items():
    PALETTE[cid] = rgb
THIN = (1, 2, 3)  # classes we care about most: conductor, structural cable, insulator

# LAS "ground_truth" (and possibly the masks) store the ORIGINAL 22 classes -> remap to 11
REMAP = np.full(256, 255, dtype=np.uint8)
for orig, train in {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3,
                    14: 4, 15: 5, 16: 6, 17: 7, 18: 7, 19: 8, 20: 9, 21: 10}.items():
    REMAP[orig] = train


def to_training_classes(labels, what=""):
    """Original IDs use 11..21 for vegetation/ground/building; the 11-class space stops at 10."""
    u = np.unique(labels)
    if ((u >= 11) & (u <= 21)).any():
        print(f"[info] {what}: original 22-class IDs detected -> remapped to 11 classes")
        return REMAP[labels]
    print(f"[info] {what}: already in the 11-class space")
    return labels


# ----------------------------------------------------------------------------- loading
def load_calib(path):
    root = ET.parse(path).getroot()

    def g(tag, default=0.0):
        el = root.find(tag)
        return float(el.text) if el is not None and el.text else default

    cal = {k: g(k) for k in ["f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "k5",
                             "p1", "p2", "p3", "p4"]}
    cal["width"], cal["height"] = int(g("width")), int(g("height"))
    return cal


def load_poses(path):
    """cameras_pose.txt: PhotoID, X, Y, Z, Omega, Phi, Kappa, r11..r33"""
    poses = {}
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t or t[0].startswith("#") or len(t) < 16:
                continue
            v = np.array(t[1:16], dtype=np.float64)
            poses[t[0]] = {"S": v[0:3], "opk": v[3:6], "R": v[6:15].reshape(3, 3)}
    return poses


def rot_from_opk(omega, phi, kappa):
    """R = Rz(kappa) Ry(phi) Rx(omega), angles in degrees (paper's convention)."""
    o, p, k = np.radians([omega, phi, kappa])
    Rx = np.array([[1, 0, 0], [0, np.cos(o), np.sin(o)], [0, -np.sin(o), np.cos(o)]])
    Ry = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(k), np.sin(k), 0], [-np.sin(k), np.cos(k), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_points(zone, max_points=0, seed=0):
    """Read the zone LAS once (chunked, low memory); cache as .npz next to it for fast reloads."""
    lidar_dir = os.path.join(zone, "lidar")
    las_files = [f for f in os.listdir(lidar_dir) if f.lower().endswith((".las", ".laz"))]
    if not las_files:
        sys.exit(f"no .las in {lidar_dir}")
    las_path = os.path.join(lidar_dir, las_files[0])
    cache = las_path.rsplit(".", 1)[0] + "_cache.npz"
    xyz = labels = None
    if os.path.exists(cache):
        try:
            with np.load(cache) as d:
                xyz, labels = d["xyz"], d["labels"]
            print(f"[info] loaded cache {cache}: {len(labels):,} points")
        except Exception as e:
            print(f"[warn] cache unreadable ({e}); rebuilding it from the LAS")
            os.remove(cache)
    if xyz is None:
        import laspy
        with laspy.open(las_path) as reader:
            n = reader.header.point_count
            xyz = np.empty((n, 3), dtype=np.float64)
            labels = np.empty(n, dtype=np.uint8)
            field = None
            done = 0
            for chunk in reader.chunk_iterator(5_000_000):
                if field is None:
                    field = "ground_truth" if "ground_truth" in \
                        chunk.point_format.extra_dimension_names else "classification"
                k = len(chunk)
                xyz[done:done + k, 0] = chunk.x
                xyz[done:done + k, 1] = chunk.y
                xyz[done:done + k, 2] = chunk.z
                labels[done:done + k] = np.asarray(chunk[field]).astype(np.uint8)
                done += k
                print(f"\r[info] reading {las_path}: {done:,}/{n:,}", end="", flush=True)
        print(f"\n[info] label field '{field}'")
        tmp = cache + ".tmp.npz"
        np.savez(tmp, xyz=xyz, labels=labels)
        os.replace(tmp, cache)
        print(f"[info] wrote cache {cache}")
    labels = to_training_classes(labels, "LAS labels")
    if max_points and len(labels) > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(labels), max_points, replace=False)
        xyz, labels = xyz[sel], labels[sel]
        print(f"[info] subsampled to {max_points:,} points")
    return xyz, labels


def load_mask(zone, name):
    p = os.path.join(zone, "masks", name + ".png")
    m = np.array(Image.open(p))
    if m.ndim == 3:
        m = m[..., 0]
    return to_training_classes(m.astype(np.uint8), "mask " + name)


def pick_image(zone, poses, stride=4, cls=3):
    """Pick the image whose mask contains the most pixels of class `cls` (default insulator)."""
    names = sorted(poses.keys())[::stride]
    best, best_n = None, -1
    for n in names:
        p = os.path.join(zone, "masks", n + ".png")
        if not os.path.exists(p):
            continue
        m = np.array(Image.open(p))
        m = m[..., 0] if m.ndim == 3 else m
        u = np.unique(m)
        if ((u >= 11) & (u <= 21)).any():
            m = REMAP[m.astype(np.uint8)]
        cnt = int((m == cls).sum())
        if cnt > best_n:
            best, best_n = n, cnt
    print(f"[info] auto-picked {best} ({best_n:,} pixels of class {cls} in its mask)")
    return best


# ----------------------------------------------------------------------------- projection
def project(xyz, R, S, cal, conv=(1, 1, 1)):
    """Return u, v (pixels), z (depth, m) and ok (in front of camera and inside the frame)."""
    sx, sy, dz = conv
    P = (xyz - S) @ R.T                      # camera coordinates, row-wise
    Zc = P[:, 2]
    z = dz * (-Zc)
    front = z > 1e-6
    x = np.zeros(len(z)); y = np.zeros(len(z))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x[front] = sx * (-P[front, 0] / Zc[front])
        y[front] = sy * (-P[front, 1] / Zc[front])
        # the distortion polynomial is only valid inside the calibrated field of view;
        # far off-axis points would otherwise be folded back into the frame
        xmax = 1.15 * (cal["width"] / 2) / cal["f"]
        ymax = 1.15 * (cal["height"] / 2) / cal["f"]
        front &= (np.abs(x) <= xmax) & (np.abs(y) <= ymax)
        x[~front] = 0; y[~front] = 0
        rc = x * x + y * y
        dr = 1 + cal["k1"] * rc + cal["k2"] * rc ** 2 + cal["k3"] * rc ** 3 \
            + cal["k4"] * rc ** 4 + cal["k5"] * rc ** 5
        tg = 1 + cal["p3"] * rc + cal["p4"] * rc ** 2
        xp = x * dr + cal["p1"] * (rc + 2 * x * x) + 2 * cal["p2"] * x * y * tg
        yp = y * dr + cal["p2"] * (rc + 2 * y * y) + 2 * cal["p1"] * x * y * tg
        u = cal["width"] / 2 + cal["cx"] + xp * cal["f"] + xp * cal["b1"] + yp * cal["b2"]
        v = cal["height"] / 2 + cal["cy"] + yp * cal["f"]
    ok = front & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < cal["width"]) \
        & (v >= 0) & (v < cal["height"])
    return u, v, z, ok


def visible(u, v, z, ok, cal, ds=4, tau=0.5):
    """Depth-map visibility test (paper eq. 14-15) on a grid downscaled by `ds`."""
    W, H = cal["width"] // ds + 1, cal["height"] // ds + 1
    idx = np.where(ok)[0]
    lin = (v[idx] / ds).astype(np.int64) * W + (u[idx] / ds).astype(np.int64)
    depth = np.full(W * H, np.inf)
    np.minimum.at(depth, lin, z[idx])
    vis = np.zeros(len(u), dtype=bool)
    vis[idx[z[idx] <= depth[lin] + tau]] = True
    return vis


def agreement(labels, u, v, vis, mask):
    """Fraction of visible points whose projected pixel carries the same label in the mask.
    'all' ignores pixels the mask leaves unlabeled (255); for the thin classes a point that
    lands on an unlabeled pixel (e.g. a cable projected onto the sky) counts as a miss."""
    ui, vi = u[vis].astype(np.int64), v[vis].astype(np.int64)
    m, L = mask[vi, ui], labels[vis]
    valid = (m != 255) & (L != 255)
    out = {"all": float((m[valid] == L[valid]).mean()) if valid.any() else 0.0,
           "n": int(valid.sum())}
    for c in THIN:
        sel = L == c
        out[c] = float((m[sel] == c).mean()) if sel.any() else float("nan")
    return out


def frustum_subset(xyz, labels, R, S, cal, conv, margin=0.3):
    """Keep only points that could fall into (an enlarged) frame: speeds up perturbation."""
    u, v, z, _ = project(xyz, R, S, cal, conv)
    W, H = cal["width"], cal["height"]
    keep = (z > 0) & (u > -margin * W) & (u < (1 + margin) * W) \
        & (v > -margin * H) & (v < (1 + margin) * H)
    return xyz[keep], labels[keep]


def rodrigues(axis, deg):
    a = axis / np.linalg.norm(axis)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


# ----------------------------------------------------------------------------- commands
def parse_conv(s):
    t = [int(x) for x in s.split(",")]
    assert len(t) == 3 and all(x in (1, -1) for x in t), "--conv must be like 1,1,1"
    return tuple(t)


def setup(args):
    cal = load_calib(os.path.join(args.zone, "pose", "camera_calibration.xml"))
    poses = load_poses(os.path.join(args.zone, "pose", "cameras_pose.txt"))
    name = args.image or pick_image(args.zone, poses)
    if name not in poses:
        sys.exit(f"image {name} not in cameras_pose.txt")
    pose = poses[name]
    dR = np.abs(rot_from_opk(*pose["opk"]) - pose["R"]).max()
    print(f"[info] image {name}: R from omega/phi/kappa vs file matrix, max |diff| = {dR:.2e}")
    xyz, labels = load_points(args.zone, args.max_points)
    mask = load_mask(args.zone, name)
    return cal, name, pose, xyz, labels, mask


def cmd_check(args):
    cal, name, pose, xyz, labels, mask = setup(args)
    print(f"\n{'conv (sx,sy,dz)':<18}{'visible':>10}{'agree all':>11}"
          f"{'conductor':>11}{'struct.':>9}{'insulator':>11}")
    results = []
    for dz in (1, -1):
        for sx in (1, -1):
            for sy in (1, -1):
                conv = (sx, sy, dz)
                u, v, z, ok = project(xyz, pose["R"], pose["S"], cal, conv)
                if ok.sum() < 1000:
                    print(f"{str(conv):<18}{int(ok.sum()):>10}   (almost nothing in frame)")
                    continue
                vis = visible(u, v, z, ok, cal, args.ds, args.tau)
                a = agreement(labels, u, v, vis, mask)
                print(f"{str(conv):<18}{int(vis.sum()):>10}{a['all']*100:>10.1f}%"
                      f"{a[1]*100:>10.1f}%{a[2]*100:>8.1f}%{a[3]*100:>10.1f}%")
                results.append((conv, int(vis.sum()), a))
    # score = mean of overall agreement and the thin-class rates; ignore variants that put
    # far fewer points in the frame than the best one (wrong depth sign)
    max_vis = max(r[1] for r in results)
    best, best_s = None, -1
    for conv, nvis, a in results:
        if nvis < 0.5 * max_vis:
            continue
        s = np.nanmean([a["all"], a[1], a[2], a[3]])
        if s > best_s:
            best, best_s = conv, s
    print(f"\n[result] best convention: --conv {best[0]},{best[1]},{best[2]}  "
          f"(score {best_s*100:.1f}%). (1,1,1) is the paper's formula as written.")


def cmd_overlay(args):
    cal, name, pose, xyz, labels, mask = setup(args)
    conv = parse_conv(args.conv)
    u, v, z, ok = project(xyz, pose["R"], pose["S"], cal, conv)
    vis = visible(u, v, z, ok, cal, args.ds, args.tau)
    a = agreement(labels, u, v, vis, mask)
    print(f"[info] visible points: {int(vis.sum()):,}  agreement with mask: {a['all']*100:.1f}%  "
          f"(conductor {a[1]*100:.1f}%, structural {a[2]*100:.1f}%, insulator {a[3]*100:.1f}%)")

    s = args.scale
    img = Image.open(os.path.join(args.zone, "images", name + ".JPG"))
    img = np.array(img.resize((cal["width"] // s, cal["height"] // s), Image.BILINEAR))
    over = img.copy()
    ui, vi = (u[vis] / s).astype(np.int64), (v[vis] / s).astype(np.int64)
    col = PALETTE[labels[vis]]
    # draw thin classes last so they stay on top
    order = np.argsort(np.isin(labels[vis], THIN).astype(np.int8), kind="stable")
    over[vi[order], ui[order]] = col[order]
    mask_rgb = PALETTE[mask[::s, ::s]][: over.shape[0], : over.shape[1]]
    panel = np.concatenate([over, mask_rgb], axis=1)

    # caption strip with the agreement numbers, so the figure explains itself
    from PIL import ImageDraw, ImageFont
    strip_h = 70
    canvas = Image.new("RGB", (panel.shape[1], panel.shape[0] + strip_h), (255, 255, 255))
    canvas.paste(Image.fromarray(panel), (0, strip_h))
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    d = ImageDraw.Draw(canvas)
    left = "LEFT: drone image + 3D LiDAR labels projected by our pipeline"
    right = "RIGHT: authors' 2D mask (reference)"
    d.text((15, 8), left, fill=(0, 0, 0), font=font)
    d.text((panel.shape[1] // 2 + 15, 8), right, fill=(0, 0, 0), font=font)
    d.text((15, 38), f"agreement with reference: {a['all']*100:.1f}%   "
                     f"(conductor {a[1]*100:.1f}%, structural cable {a[2]*100:.1f}%, "
                     f"insulator {a[3]*100:.1f}%)   zone {os.path.basename(args.zone)} / {name}",
           fill=(0, 0, 0), font=font)
    canvas.save(args.out)
    print(f"[info] wrote {args.out}  (left: image + projected 3D labels, right: provided 2D mask)")


def cmd_perturb(args):
    cal, name, pose, xyz, labels, mask = setup(args)
    conv = parse_conv(args.conv)
    R0, S0 = pose["R"], pose["S"]
    xyz, labels = frustum_subset(xyz, labels, R0, S0, cal, conv)
    print(f"[info] points near the frustum: {len(labels):,}")
    rng = np.random.default_rng(args.seed)

    def hit_rates(R, S):
        u, v, z, ok = project(xyz, R, S, cal, conv)
        vis = visible(u, v, z, ok, cal, args.ds, args.tau)
        return agreement(labels, u, v, vis, mask)

    rows = []
    base = hit_rates(R0, S0)
    rows.append(("none", 0.0, base))
    print(f"\n{'perturbation':<14}{'level':>8}{'all':>8}{'conductor':>11}{'struct.':>9}{'insulator':>11}")

    def show(kind, level, a):
        print(f"{kind:<14}{level:>8.2f}{a['all']*100:>7.1f}%{a[1]*100:>10.1f}%"
              f"{a[2]*100:>8.1f}%{a[3]*100:>10.1f}%")

    show("none", 0, base)
    for deg in [float(x) for x in args.rot.split(",") if x]:
        acc = []
        for _ in range(args.trials):
            Rd = rodrigues(rng.normal(size=3), deg)
            acc.append(hit_rates(Rd @ R0, S0))
        a = {k: float(np.nanmean([x[k] for x in acc])) for k in acc[0]}
        rows.append(("rotation_deg", deg, a)); show("rotation_deg", deg, a)
    for cm in [float(x) for x in args.trans.split(",") if x]:
        acc = []
        for _ in range(args.trials):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            acc.append(hit_rates(R0, S0 + d * cm / 100.0))
        a = {k: float(np.nanmean([x[k] for x in acc])) for k in acc[0]}
        rows.append(("translation_cm", cm, a)); show("translation_cm", cm, a)

    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["perturbation", "level", "all", "conductor", "structural", "insulator", "n"])
        for kind, level, a in rows:
            w.writerow([kind, level, a["all"], a[1], a[2], a[3], a["n"]])

    import os as _os
    _os.environ["MPLBACKEND"] = "Agg"     # a notebook kernel exports its inline backend; a plain subprocess cannot load it
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, kind, xlabel in [(axes[0], "rotation_deg", "camera rotation error (deg)"),
                             (axes[1], "translation_cm", "camera position error (cm)")]:
        sub = [(lv, a) for k, lv, a in rows if k == kind]
        xs = [0.0] + [lv for lv, _ in sub]
        for key, lab in [("all", "all classes"), (1, "conductor cable"),
                         (2, "structural cable"), (3, "insulator")]:
            ys = [base[key] * 100] + [a[key] * 100 for _, a in sub]
            ax.plot(xs, ys, marker="o", label=lab)
        ax.set_xlabel(xlabel); ax.set_ylabel("points landing on correct mask label (%)")
        ax.set_xscale("symlog", linthresh=min(xs[1:]) if len(xs) > 1 else 1)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"GridNet-HD {os.path.basename(args.zone)} / {name}: sensitivity to pose error")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    print(f"[info] wrote {args.out}.csv and {args.out}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--zone", required=True, help="zone folder, e.g. data/t5b5")
    common.add_argument("--image", default=None, help="photo id without extension")
    common.add_argument("--auto", action="store_true", help="auto-pick the image with most insulator pixels")
    common.add_argument("--max-points", type=int, default=0, help="random subsample of the LAS (0 = all)")
    common.add_argument("--ds", type=int, default=4, help="depth-map downscale factor")
    common.add_argument("--tau", type=float, default=0.5, help="visibility depth tolerance (m)")
    common.add_argument("--conv", default="1,1,1", help="sign convention sx,sy,dz (see `check`)")
    p = sub.add_parser("check", parents=[common]); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("overlay", parents=[common]); p.set_defaults(fn=cmd_overlay)
    p.add_argument("--out", default="overlay.png"); p.add_argument("--scale", type=int, default=4)
    p = sub.add_parser("perturb", parents=[common]); p.set_defaults(fn=cmd_perturb)
    p.add_argument("--out", default="perturb"); p.add_argument("--rot", default="0.05,0.1,0.25,0.5,1")
    p.add_argument("--trans", default="2,5,10,25,50"); p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.image and not args.auto:
        sys.exit("give --image NAME or --auto")
    args.fn(args)


if __name__ == "__main__":
    main()
