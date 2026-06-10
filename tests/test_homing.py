import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))
import numpy as np
from robotgeometry import FiveBarGeometry
from ikine import ik, working_modes
from fkine import fk
from homing import home, calibrate_offsets

G = FiveBarGeometry(0.20, 0.24, 0.26, 0.32, 0.20)


def test_home_identifies_branches():
    for om in [(+1, -1), (-1, +1), (+1, +1)]:
        q = ik(G, [0.01, 0.28], om)
        for sigma in (+1, -1):
            Xs = fk(G, q, sigma)
            h = home(G, q, Xs + np.array([0.002, -0.001]))
            assert h.sigma == sigma
            assert h.omega == working_modes(G, q, Xs)
            assert np.allclose(h.X, Xs, atol=1e-12)
            assert (h.det_A > 0) == (sigma > 0)


def test_calibrate_offsets_recovers_truth():
    true_off = np.array([0.013, -0.021])
    qs, Xs = [], []
    for Xk in ([0.00, 0.30], [0.06, 0.25], [-0.05, 0.33]):
        qs.append(ik(G, Xk, (+1, -1)) - true_off)
        Xs.append(Xk)
    res = calibrate_offsets(G, qs, Xs)
    assert res.converged
    assert np.allclose(res.offsets, true_off, atol=1e-10)
    assert res.rms_residual < 1e-12


def test_calibrate_with_noise():
    rng = np.random.default_rng(4)
    true_off = np.array([-0.008, 0.017])
    qs, Xs = [], []
    for _ in range(8):
        Xk = np.array([rng.uniform(-0.06, 0.06), rng.uniform(0.24, 0.34)])
        qs.append(ik(G, Xk, (+1, -1)) - true_off + rng.normal(0, 1e-4, 2))
        Xs.append(Xk)
    res = calibrate_offsets(G, qs, Xs)
    assert np.allclose(res.offsets, true_off, atol=2e-3)
