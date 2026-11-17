"""Metashape frame-camera model for GridNet-HD (paper supplementary Sec. 9) + visibility test.

Validated on t5b5 against the authors' 2D masks: the sign convention (sx, sy, dz) = (1, -1, 1)
reproduces the masks (94.5 % agreement); the paper's formula as written, (1, 1, 1), does not.
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

VALIDATED_CONV = (1, -1, 1)


@dataclass
class Calibration:
    width: int
    height: int
    f: float
    cx: float = 0.0
    cy: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    p3: float = 0.0
    p4: float = 0.0

    @classmethod
    def from_xml(cls, path: str) -> "Calibration":
        root = ET.parse(path).getroot()

        def g(tag, default=0.0):
            el = root.find(tag)
            return float(el.text) if el is not None and el.text else default

        return cls(width=int(g("width")), height=int(g("height")), f=g("f"),
                   **{k: g(k) for k in ["cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "k5",
                                        "p1", "p2", "p3", "p4"]})

    @property
    def half_fov_tan(self) -> Tuple[float, float]:
        return (self.width / 2) / self.f, (self.height / 2) / self.f


@dataclass
class Pose:
    S: np.ndarray      # camera centre (3,)
    R: np.ndarray      # world->camera rotation (3,3), rows = camera axes
    opk: np.ndarray    # omega, phi, kappa in degrees

    @property
    def view_dir(self) -> np.ndarray:
        """Unit viewing direction in world coordinates (camera looks along -Z_cam)."""
        return -self.R[2]


def rot_from_opk(omega: float, phi: float, kappa: float) -> np.ndarray:
    """R = Rz(kappa) Ry(phi) Rx(omega), degrees (Metashape / paper convention)."""
    o, p, k = np.radians([omega, phi, kappa])
    Rx = np.array([[1, 0, 0], [0, np.cos(o), np.sin(o)], [0, -np.sin(o), np.cos(o)]])
    Ry = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(k), np.sin(k), 0], [-np.sin(k), np.cos(k), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_poses(path: str) -> Dict[str, Pose]:
    """cameras_pose.txt: PhotoID X Y Z Omega Phi Kappa r11..r33."""
    poses: Dict[str, Pose] = {}
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t or t[0].startswith("#") or len(t) < 16:
                continue
            v = np.array(t[1:16], dtype=np.float64)
            poses[t[0]] = Pose(S=v[0:3], R=v[6:15].reshape(3, 3), opk=v[3:6])
    return poses


def load_zone_cameras(zone_dir: str) -> Tuple[Calibration, Dict[str, Pose]]:
    cal = Calibration.from_xml(os.path.join(zone_dir, "pose", "camera_calibration.xml"))
    poses = load_poses(os.path.join(zone_dir, "pose", "cameras_pose.txt"))
    return cal, poses


def project(xyz: np.ndarray, pose: Pose, cal: Calibration, conv=VALIDATED_CONV,
            fov_margin: float = 1.15):
    """Project world points to pixels.

    Returns u, v (pixels, float64), z (depth along the optical axis, metres) and ok (bool:
    in front of the camera, inside the calibrated field of view and inside the frame)."""
    sx, sy, dz = conv
    P = (xyz - pose.S) @ pose.R.T
    Zc = P[:, 2]
    z = dz * (-Zc)
    front = z > 1e-6
    n = len(z)
    x = np.zeros(n); y = np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x[front] = sx * (-P[front, 0] / Zc[front])
        y[front] = sy * (-P[front, 1] / Zc[front])
        tx, ty = cal.half_fov_tan
        # the distortion polynomial is only valid inside the calibrated FOV; far off-axis
        # points would otherwise be folded back into the frame
        front &= (np.abs(x) <= fov_margin * tx) & (np.abs(y) <= fov_margin * ty)
        x[~front] = 0; y[~front] = 0
        rc = x * x + y * y
        dr = 1 + cal.k1 * rc + cal.k2 * rc ** 2 + cal.k3 * rc ** 3 + cal.k4 * rc ** 4 + cal.k5 * rc ** 5
        tg = 1 + cal.p3 * rc + cal.p4 * rc ** 2
        xp = x * dr + cal.p1 * (rc + 2 * x * x) + 2 * cal.p2 * x * y * tg
        yp = y * dr + cal.p2 * (rc + 2 * y * y) + 2 * cal.p1 * x * y * tg
        u = cal.width / 2 + cal.cx + xp * cal.f + xp * cal.b1 + yp * cal.b2
        v = cal.height / 2 + cal.cy + yp * cal.f
    ok = front & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < cal.width) & (v >= 0) & (v < cal.height)
    return u, v, z, ok


def depth_map(u, v, z, ok, cal: Calibration, ds: int = 4):
    """Min-depth map on a grid downscaled by `ds` (paper eq. 14). Returns (depth, W, H)."""
    W, H = cal.width // ds + 1, cal.height // ds + 1
    idx = np.where(ok)[0]
    lin = (v[idx] / ds).astype(np.int64) * W + (u[idx] / ds).astype(np.int64)
    depth = np.full(W * H, np.inf)
    np.minimum.at(depth, lin, z[idx])
    return depth, W, H


def visible(u, v, z, ok, cal: Calibration, ds: int = 4, tau: float = 0.5):
    """Visibility test (paper eq. 15). Returns vis (bool, all points) and depth residual
    z - depthmap (metres, 0 for the nearest surface) for the ok points."""
    depth, W, _ = depth_map(u, v, z, ok, cal, ds)
    idx = np.where(ok)[0]
    lin = (v[idx] / ds).astype(np.int64) * W + (u[idx] / ds).astype(np.int64)
    resid = z[idx] - depth[lin]
    vis = np.zeros(len(u), dtype=bool)
    vis[idx[resid <= tau]] = True
    resid_full = np.zeros(len(u), dtype=np.float32)
    resid_full[idx] = resid
    return vis, resid_full


def frustum_prefilter(xyz: np.ndarray, pose: Pose, cal: Calibration, max_dist: float = 150.0,
                      margin: float = 1.3) -> np.ndarray:
    """Cheap bounding test before projecting: in front of the camera, within max_dist and
    inside an enlarged FOV cone. Returns indices."""
    d = xyz - pose.S
    dist = np.linalg.norm(d, axis=1)
    zc = d @ (-pose.R[2])          # depth along viewing direction
    cand = (zc > 0.5) & (dist < max_dist)
    tx, ty = cal.half_fov_tan
    half_tan = margin * np.hypot(tx, ty)
    lateral = np.sqrt(np.maximum(dist[cand] ** 2 - zc[cand] ** 2, 0))
    keep = np.where(cand)[0][lateral <= half_tan * zc[cand]]
    return keep


def rodrigues(axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def perturb_pose(pose: Pose, rng: np.random.Generator, rot_deg: float = 0.0, trans_m: float = 0.0) -> Pose:
    """Random pose perturbation: rotation about a random axis by rot_deg, translation of
    length trans_m in a random direction (the i.i.d. noise model of the C2 protocol)."""
    R, S = pose.R, pose.S
    if rot_deg > 0:
        R = rodrigues(rng.normal(size=3), rot_deg) @ R
    if trans_m > 0:
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        S = S + d * trans_m
    return Pose(S=S, R=R, opk=pose.opk)


def perturb_pose_gaussian(pose: Pose, rng: np.random.Generator, sigma_t: float = 0.0, sigma_r_deg: float = 0.0) -> Pose:
    """The i.i.d. error model of the robustness protocol for one camera (numpy twin of
    `gridnet.train.fusion.perturb_cams`): camera centre + N(0, sigma_t^2 I) metres, rotation by the axis-angle
    vector w with per-axis angles ~ N(0, sigma_r^2) degrees (angle |w| about w / |w|)."""
    R, S = pose.R, pose.S
    if sigma_t > 0:
        S = S + rng.normal(size=3) * sigma_t
    if sigma_r_deg > 0:
        w = rng.normal(size=3) * sigma_r_deg
        R = rodrigues(w, float(np.linalg.norm(w))) @ R
    return Pose(S=S, R=R, opk=pose.opk)
