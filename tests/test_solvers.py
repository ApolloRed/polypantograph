import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))
import numpy as np
from robotgeometry import FiveBarGeometry
from ikine import ik, working_modes
from jacobians import jacobian
from ikine import (ClosedFormSolver, TaskNewtonSolver,
                   PerLegNewtonSolver)
from fkine import FKNewtonEstimator

G = FiveBarGeometry(0.20, 0.24, 0.26, 0.32, 0.20)
CENTER = np.array([0.02, 0.30])
OMEGA = working_modes(G, ik(G, CENTER, (+1, -1)), CENTER)


def _circle(t):
    th = 2 * np.pi * t
    return CENTER + 0.05 * np.array([np.cos(th), np.sin(th)])


def test_three_solvers_agree_on_path():
    sa = ClosedFormSolver(G, OMEGA)
    sb = TaskNewtonSolver(G, +1, eps=1e-11)
    sc = PerLegNewtonSolver(G, eps=1e-11)
    q = ik(G, _circle(0.0), OMEGA)
    qa = qb = qc = q
    for k in range(1, 400):
        X = _circle(k / 400)
        ra = sa.solve(X, qa); rb = sb.solve(X, qb); rc = sc.solve(X, qc)
        qa, qb, qc = ra.q, rb.q, rc.q
        assert not (ra.flag or rb.flag or rc.flag)
        assert np.allclose(qa, qb, atol=1e-8)
        assert np.allclose(qa, qc, atol=1e-8)
        assert rb.residual <= 1e-9 and rc.residual <= 1e-9


def test_closed_form_guard():
    s = ClosedFormSolver(G, OMEGA, dq_guard=0.05)
    q0 = ik(G, CENTER, OMEGA)
    r = s.solve(CENTER + np.array([0.10, 0.0]), q0)
    assert r.flag and np.allclose(r.q, q0)


def test_per_leg_damped_near_boundary():
    s = PerLegNewtonSolver(G, eps=1e-9, n_max=15, mu=5e-3)
    rmax = G.reach_annulus(0)[1]
    X = G.bases[0] + (rmax - 1e-3) * np.array([np.cos(1.0), np.sin(1.0)])
    q0 = ik(G, X - 0.002 * np.array([np.cos(1.0), np.sin(1.0)]), OMEGA)
    assert s.solve(X, q0).residual < 1e-6


def test_fk_estimator_bound():
    est = FKNewtonEstimator(G, sigma0=+1)
    res = 2 * np.pi / 2 ** 16
    est.reset(np.round(ik(G, _circle(0.0), OMEGA) / res) * res)
    for k in range(300):
        X_true = _circle(k / 300)
        q = ik(G, X_true, OMEGA)
        out = est.update(np.round(q / res) * res)
        bound = np.linalg.svd(jacobian(G, q, X_true), compute_uv=False)[0] * res
        assert np.linalg.norm(out.X - X_true) <= bound + 1e-9
        assert not out.flag
        assert out.inv_cond > 0.0 and out.det_A > 0.0
