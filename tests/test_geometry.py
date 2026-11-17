import numpy as np
from gridnet.preprocess.geometry import pca_features, voxel_grid, majority_labels, height_above_ground


def test_pca_shapes_are_discriminative():
    rng = np.random.default_rng(0)
    line = np.column_stack([np.linspace(0, 20, 4000), np.zeros(4000), np.full(4000, 10.0)]) + rng.normal(0, 0.005, (4000, 3))
    ground = np.column_stack([rng.uniform(0, 20, 20000), rng.uniform(0, 20, 20000), np.zeros(20000)]) + rng.normal(0, 0.01, (20000, 3))
    wall = np.column_stack([np.full(8000, 5.0), rng.uniform(0, 5, 8000), rng.uniform(0, 5, 8000)]) + rng.normal(0, 0.01, (8000, 3))
    xyz = np.concatenate([line, ground, wall])
    f = pca_features(xyz, k=16)
    n_l, n_g = len(line), len(ground)
    assert f["linearity"][:n_l].mean() > 0.9
    # with k=16 random samples on a plane l1 != l2, so planarity sits around 0.5; the
    # discriminative signals are sphericity ~0 and linearity well below the line's
    assert f["planarity"][n_l:n_l + n_g].mean() > 0.4
    assert f["sphericity"][n_l:n_l + n_g].mean() < 0.05
    assert f["linearity"][n_l:n_l + n_g].mean() < 0.6
    assert f["verticality"][n_l:n_l + n_g].mean() < 0.1          # horizontal ground
    assert f["verticality"][n_l + n_g:].mean() > 0.9             # vertical wall
    hag = height_above_ground(xyz, cell=1.0)
    assert abs(hag[:n_l].mean() - 10.0) < 0.5
    assert hag[n_l:n_l + n_g].mean() < 0.2


def test_voxel_grid_and_majority():
    xyz = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [1, 1, 1]], dtype=float)
    labels = np.array([1, 1, 2, 255], dtype=np.uint8)
    keep, inv = voxel_grid(xyz, 0.1)
    assert len(keep) == 2 and inv[0] == inv[1] == inv[2] and inv[3] != inv[0]
    maj = majority_labels(labels, inv, len(keep))
    assert maj[inv[0]] == 1 and maj[inv[3]] == 255
