import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))
import numpy as np
import pytest
from robotgeometry import FiveBarGeometry
from dynamics import DynamicParameters, FiveBarDynamics
from jacobians import jacobian
from fkine import fk
from ikine import ik

GEOM = FiveBarGeometry.symmetric(0.20, 0.25, 0.20)
PAR = DynamicParameters.uniform(GEOM, 0.5, 0.3, apex_mass=0.10)
Kd = np.array([800.0, 500.0])
Dd = np.array([12.0, 8.0])
OMEGA, SIGMA, RATE = (+1, -1), +1, 1000.0


def _impedance_sim(dyn, Xd, force, window, T):
    """Integrate impedance control tau = g + J^T(Kd e + Dd edot) holding Xd,
    with ``force`` applied at the apex during ``window`` = (t0, t1).  Returns
    the time history of end-effector position, joint angles, and error norm."""
    q = ik(GEOM, Xd, OMEGA)
    qd = np.zeros(2)
    n = int(T * RATE)
    X_hist = np.empty((n, 2)); q_hist = np.empty((n, 2)); e_hist = np.empty(n)
    for k in range(n):
        tk = k / RATE
        X = fk(GEOM, q, SIGMA)
        J = jacobian(GEOM, q, X)
        Xdot = J @ qd
        tau = dyn.gravity_torque(q) + J.T @ (Kd * (Xd - X) + Dd * (-Xdot))
        Fp = force if (window[0] <= tk < window[1]) else np.zeros(2)
        qd = qd + dyn.forward_dynamics(q, qd, tau, wrench=Fp) / RATE
        q = q + qd / RATE
        X_hist[k] = X; q_hist[k] = q; e_hist[k] = np.linalg.norm(Xd - X)
    return X_hist, q_hist, e_hist


@pytest.mark.parametrize("gravity", [(0.0, 0.0), (0.0, -9.81)])
def test_impedance_steady_state_compliance(gravity):
    """At static equilibrium the rendered stiffness gives X - Xd = Kd^{-1} F,
    independent of the link inertia and (with gravity compensation) of the
    mounting."""
    dyn = FiveBarDynamics(GEOM, PAR, gravity=gravity)
    Xd = np.array([0.0, 0.30])
    F = np.array([5.0, -4.0])
    X_hist, _, _ = _impedance_sim(dyn, Xd, F, window=(0.0, 1e9), T=4.0)
    X = X_hist[-1]
    assert np.allclose(X - Xd, F / Kd, atol=1e-5)


def test_impedance_regulation_no_disturbance():
    """With no disturbance the end-effector stays on the held reference."""
    dyn = FiveBarDynamics(GEOM, PAR, gravity=(0.0, -9.81))
    Xd = np.array([0.0, 0.30])
    _, _, e = _impedance_sim(dyn, Xd, np.zeros(2), window=(0.0, 0.0), T=1.0)
    assert e.max() < 1e-9


def test_impedance_springback_after_release():
    """A step force deflects the apex; after release it returns to the
    reference (the spring-back of the rendered impedance)."""
    dyn = FiveBarDynamics(GEOM, PAR, gravity=(0.0, -9.81))
    Xd = np.array([0.0, 0.30])
    F = np.array([5.0, -4.0])
    _, _, e = _impedance_sim(dyn, Xd, F, window=(1.0, 1.3), T=3.0)
    push = e[int(1.0 * RATE):int(1.3 * RATE)]
    assert push.max() > 5e-3           # it actually deflected
    assert e[-1] < 1e-4                # and came back
