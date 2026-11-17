import numpy as np
from gridnet.camera import Calibration, Pose, rot_from_opk, project, perturb_pose


def _cam():
    cal = Calibration(width=5280, height=3956, f=3714.63, cx=-10.87, cy=-48.48, k1=-0.088, k2=-0.098, k3=0.155, k4=-0.098)
    R = rot_from_opk(40.37, -9.09, -9.89)
    return cal, Pose(S=np.array([0.0, 0.0, 50.0]), R=R, opk=np.array([40.37, -9.09, -9.89]))


def test_principal_ray_hits_image_centre():
    cal, pose = _cam()
    p = pose.S + 40.0 * pose.view_dir          # point 40 m straight ahead
    u, v, z, ok = project(p[None], pose, cal)
    assert ok[0] and abs(z[0] - 40) < 1e-6
    assert abs(u[0] - (cal.width / 2 + cal.cx)) < 1e-3 and abs(v[0] - (cal.height / 2 + cal.cy)) < 1e-3


def test_points_behind_camera_rejected():
    cal, pose = _cam()
    p = pose.S - 40.0 * pose.view_dir
    assert not project(p[None], pose, cal)[3][0]


def test_perturbation_magnitude():
    cal, pose = _cam()
    rng = np.random.default_rng(0)
    q = perturb_pose(pose, rng, rot_deg=0.1, trans_m=0.10)
    assert abs(np.linalg.norm(q.S - pose.S) - 0.10) < 1e-9
    ang = np.degrees(np.arccos(np.clip((np.trace(q.R @ pose.R.T) - 1) / 2, -1, 1)))
    assert abs(ang - 0.1) < 1e-6
