import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))
import numpy as np
import pytest
from robotgeometry import FiveBarGeometry
from dynamics import DynamicParameters, FiveBarDynamics
from fkine import fk
from ikine import ik

# A deliberately GENERAL robot: the two legs have different link lengths
# (asymmetric), and the inertial parameters are non-uniform -- centre-of-mass
# offsets are not L/2 and inertias are not m L^2 / 12, and they differ leg to
# leg.  This exercises the code paths that the symmetric/uniform fixtures do
# not.
GEOM = FiveBarGeometry(l1=0.18, l2=0.22, l3=0.23, l4=0.27, d=0.19)
PAR = DynamicParameters(proximal_mass=[0.45, 0.55], proximal_com=[0.07, 0.12],
                        proximal_inertia=[0.0012, 0.0020],
                        distal_mass=[0.28, 0.34], distal_com=[0.10, 0.16],
                        distal_inertia=[0.0011, 0.0019], apex_mass=0.08)

# assembly mode consistent with the working modes used for sampling
_X0 = np.array([0.0, 0.30]); _q0 = ik(GEOM, _X0, (+1, -1))
SIGMA = +1 if np.linalg.norm(fk(GEOM, _q0, +1) - _X0) < 1e-9 else -1


def _samples(n, seed):
    """Reachable joint configurations: sample apex points, invert, and keep
    those whose assembly mode is SIGMA."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        X = np.array([rng.uniform(-0.05, 0.05), rng.uniform(0.27, 0.33)])
        try:
            q = ik(GEOM, X, (+1, -1))
            if np.linalg.norm(fk(GEOM, q, SIGMA) - X) < 1e-9:
                out.append(q)
        except Exception:
            pass
    return out


def test_setup_is_asymmetric_and_nonuniform():
    """Guard: the fixture really is general, not symmetric/uniform."""
    assert GEOM.l1 != GEOM.l2 and GEOM.l3 != GEOM.l4          # asymmetric
    assert PAR.proximal_mass[0] != PAR.proximal_mass[1]       # legs differ
    # centre of mass not at the midpoint, inertia not the slender-rod value
    assert not np.isclose(PAR.proximal_com[0], GEOM.l1 / 2)
    assert not np.isclose(PAR.distal_inertia[1],
                          PAR.distal_mass[1] * GEOM.l4 ** 2 / 12)


@pytest.mark.parametrize("gravity", [(0.0, 0.0), (0.0, -9.81)])
def test_mass_matrix_spd(gravity):
    dyn = FiveBarDynamics(GEOM, PAR, gravity=gravity, sigma=SIGMA)
    for q in _samples(60, 1):
        M = dyn.mass_matrix(q)
        assert np.allclose(M, M.T, atol=1e-12)
        assert np.linalg.eigvalsh(M).min() > 0


def test_passivity_skew():
    dyn = FiveBarDynamics(GEOM, PAR, sigma=SIGMA)
    rng = np.random.default_rng(2)
    for q in _samples(20, 3):
        qd = rng.normal(0, 0.7, 2)
        h = 1e-6
        Mdot = (dyn.mass_matrix(q + h*qd) - dyn.mass_matrix(q - h*qd)) / (2*h)
        S = Mdot - 2*dyn.coriolis(q, qd)
        assert np.abs(S + S.T).max() < 1e-7


def test_inverse_forward_consistency():
    rng = np.random.default_rng(4)
    err = 0.0
    for gravity in [(0.0, 0.0), (0.0, -9.81)]:
        dyn = FiveBarDynamics(GEOM, PAR, gravity=gravity, sigma=SIGMA)
        for q in _samples(40, 5):
            qd = rng.normal(0, 0.6, 2); qdd = rng.normal(0, 2, 2)
            w = rng.normal(0, 5, 3)
            tau = dyn.inverse_dynamics(q, qd, qdd, wrench=w)
            err = max(err, np.linalg.norm(
                dyn.forward_dynamics(q, qd, tau, wrench=w) - qdd))
    assert err < 1e-10


def test_gravity_matches_potential_gradient():
    dyn = FiveBarDynamics(GEOM, PAR, gravity=(0.0, -9.81), sigma=SIGMA)
    h = 1e-6
    for q in _samples(20, 6):
        gnum = np.array([
            (dyn.potential_energy(q + np.array([h, 0])) -
             dyn.potential_energy(q - np.array([h, 0]))) / (2*h),
            (dyn.potential_energy(q + np.array([0, h])) -
             dyn.potential_energy(q - np.array([0, h]))) / (2*h)])
        assert np.allclose(dyn.gravity_torque(q), gnum, atol=1e-5)


def test_horizontal_gravity_zero():
    dyn = FiveBarDynamics(GEOM, PAR, gravity=(0.0, 0.0), sigma=SIGMA)
    for q in _samples(10, 7):
        assert np.allclose(dyn.gravity_torque(q), 0.0)


def test_energy_conservation_coast():
    dyn = FiveBarDynamics(GEOM, PAR, gravity=(0.0, 0.0), sigma=SIGMA)
    q = _q0.copy(); qd = np.array([0.8, -0.6])
    st = np.concatenate([q, qd]); dt = 2e-4
    E0 = dyn.kinetic_energy(q, qd)

    def deriv(s):
        return np.concatenate([s[2:], dyn.forward_dynamics(s[:2], s[2:],
                                                            np.zeros(2))])
    for _ in range(2000):                # 0.4 s
        k1 = deriv(st); k2 = deriv(st+dt/2*k1)
        k3 = deriv(st+dt/2*k2); k4 = deriv(st+dt*k3)
        st = st + dt/6*(k1+2*k2+2*k3+k4)
    assert abs(dyn.kinetic_energy(st[:2], st[2:]) - E0) / E0 < 1e-8
