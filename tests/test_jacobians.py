import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))
import numpy as np
from robotgeometry import FiveBarGeometry
from ikine import ik
from fkine import fk, distal_angles
from jacobians import (jacobian, jacobian_inverse, det_A, det_B, dls_inverse,
                       inv_condition, singular_values, velocity_ik,
                       velocity_fk)

G = FiveBarGeometry(0.20, 0.24, 0.26, 0.32, 0.20)


def _config(X_target):
    q = ik(G, X_target, (+1, -1))
    return q, fk(G, q, +1)


def test_jacobian_vs_numeric():
    q, X = _config(np.array([0.02, 0.30]))
    J = jacobian(G, q, X)
    h = 1e-7
    Jn = np.zeros((2, 2))
    for j in range(2):
        dq = np.zeros(2); dq[j] = h
        Jn[:, j] = (fk(G, q + dq, +1) - fk(G, q - dq, +1)) / (2 * h)
    assert np.allclose(J, Jn, atol=1e-6)


def test_inverse_consistency_and_dets():
    q, X = _config(np.array([-0.04, 0.28]))
    J = jacobian(G, q, X)
    assert np.allclose(jacobian_inverse(G, q, X) @ J, np.eye(2), atol=1e-10)
    phi = distal_angles(G, q, X)
    assert np.isclose(det_A(G, q, X),
                      G.l3 * G.l4 * np.sin(phi[1] - phi[0]), atol=1e-12)
    assert np.isclose(np.linalg.det(J), det_B(G, q, X) / det_A(G, q, X),
                      atol=1e-10)


def test_velocity_maps():
    q, X = _config(np.array([0.05, 0.32]))
    qdot = np.array([0.4, -0.7])
    Xdot = velocity_fk(G, q, X, qdot)
    assert np.allclose(velocity_ik(G, q, X, Xdot), qdot, atol=1e-10)


def test_dls_limits():
    q, X = _config(np.array([0.02, 0.30]))
    J = jacobian(G, q, X)
    assert np.allclose(dls_inverse(J, 0.0), np.linalg.inv(J), atol=1e-12)
    assert np.allclose(dls_inverse(J, 1e-6), np.linalg.inv(J), atol=1e-3)
    mu = 0.05
    s = np.linalg.svd(dls_inverse(J, mu), compute_uv=False)
    assert s[0] <= 1.0 / (2 * mu) + 1e-9


def test_singular_values_closed_form():
    q, X = _config(np.array([-0.02, 0.35]))
    J = jacobian(G, q, X)
    assert np.allclose(singular_values(J), np.linalg.svd(J, compute_uv=False),
                       atol=1e-12)
    assert 0.0 < inv_condition(G, q, X) <= 1.0
