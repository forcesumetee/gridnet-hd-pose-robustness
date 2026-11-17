"""Create a small synthetic zone with the real t5b5 calibration and 3 nearby camera poses.
Writes a cache .npz (so laspy is not required) + masks. Usage: python tests/make_synth_zone.py <out_dir>"""
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gridnet.camera import Calibration, Pose, rot_from_opk, project, visible, load_zone_cameras
from gridnet.classes import REMAP

out = sys.argv[1]
for d in ["lidar", "pose", "masks", "images"]:
    os.makedirs(os.path.join(out, d), exist_ok=True)
open(os.path.join(out, "pose", "camera_calibration.xml"), "w").write("""<?xml version="1.0" encoding="UTF-8"?>
<calibration><projection>frame</projection><width>5280</width><height>3956</height>
<f>3714.6286388642334</f><cx>-10.866768367239905</cx><cy>-48.478210074502343</cy>
<b1>-0.0065770074701711739</b1><b2>-0.31007555530655018</b2>
<k1>-0.088278358429855347</k1><k2>-0.098274437413037097</k2><k3>0.15525328696222979</k3><k4>-0.098221189761332775</k4>
<p1>-0.00015959778710177182</p1><p2>-0.00019798670409981826</p2></calibration>""")
S0 = np.array([362139.0684175820788369, 5106873.6122552622109652, 1617.4670392623868338])
opk = (40.3652237454931893, -9.0893085194315617, -9.8877417780946057)
R = rot_from_opk(*opk)
rows = ["# Cameras (3)", "# PhotoID, X, Y, Z, Omega, Phi, Kappa, r11..r33"]
poses = {}
for i, shift in enumerate([(0, 0, 0), (0.7, 3.0, 1.0), (2.5, 10.0, 3.5)]):
    S = S0 + np.array(shift)
    pid = f"IMG_{i:04d}"
    poses[pid] = Pose(S=S, R=R, opk=np.array(opk))
    rows.append(pid + "\t" + "\t".join(f"{v:.10f}" for v in [*S, *opk, *R.ravel()]))
open(os.path.join(out, "pose", "cameras_pose.txt"), "w").write("\n".join(rows) + "\n")

rng = np.random.default_rng(1)
def cam_to_world(x, y, z):       # x=-Xc/Zc, y=+? ; use validated conv (1,-1,1): y_img = -(-Yc/Zc)
    Pc = np.column_stack([x * z, -y * z, -z])
    return Pc @ R + S0
pts, lab = [], []
def add(n, x, y, z, c):
    pts.append(cam_to_world(x, y, z)); lab.append(np.full(n, c, np.uint8))
n = 800_000
add(n, rng.uniform(-0.75, 0.75, n), rng.uniform(-0.55, 0.55, n), rng.uniform(45, 60, n), 16)   # herbaceous
m = 40_000; add(m, rng.uniform(-0.75, 0.75, m), 0.10 + rng.normal(0, 0.002, m), rng.uniform(30, 32, m), 5)   # conductor
m = 40_000; add(m, rng.uniform(-0.75, 0.75, m), -0.20 + rng.normal(0, 0.002, m), rng.uniform(30, 32, m), 6)  # guard cable
m = 20_000; add(m, rng.normal(0.3, 0.01, m), rng.normal(0.1, 0.01, m), rng.uniform(31, 32, m), 10)          # insulator
m = 150_000; add(m, rng.uniform(-0.6, -0.3, m), rng.uniform(-0.4, -0.1, m), rng.uniform(25, 28, m), 14)      # tree
m = 100_000; add(m, rng.uniform(-1.5, 1.5, m), rng.uniform(-1.2, 1.2, m), rng.uniform(-40, -5, m), 16)       # behind camera
xyz = np.concatenate(pts); labels = np.concatenate(lab)
open(os.path.join(out, "lidar", "synth.las"), "wb").close()
np.savez(os.path.join(out, "lidar", "synth_cache.npz"), xyz=xyz, labels_raw=labels)

cal, _ = load_zone_cameras(out)
for pid, pose in poses.items():
    u, v, z, ok = project(xyz, pose, cal)
    vis, _ = visible(u, v, z, ok, cal)
    mask = np.full((cal.height, cal.width), 255, np.uint8)
    ui, vi = u[vis].astype(int), v[vis].astype(int)
    L = labels[vis]
    order = np.argsort(np.isin(REMAP[L], (1, 2, 3)).astype(np.int8), kind="stable")
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            mask[np.clip(vi[order] + dv, 0, cal.height - 1), np.clip(ui[order] + du, 0, cal.width - 1)] = L[order]
    Image.fromarray(mask).save(os.path.join(out, "masks", pid + ".png"))
    Image.fromarray(np.full((cal.height, cal.width, 3), 90, np.uint8)).save(os.path.join(out, "images", pid + ".JPG"), quality=50)
print("synthetic zone:", len(labels), "points,", len(poses), "cameras ->", out)
