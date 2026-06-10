import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))
import numpy as np
import pytest
from robotgeometry import FiveBarGeometry
from dynamics import DynamicParameters, FiveBarDynamics
from urdf_loader import load_urdf, build_dynamics

GEOM = FiveBarGeometry.symmetric(0.20, 0.25, 0.20)
PAR = DynamicParameters.uniform(GEOM, 0.5, 0.3, apex_mass=0.10)
URDF = os.path.join(os.path.dirname(__file__), "..", "examples", "fivebar.urdf")


def _samples(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        q = np.array([rng.uniform(1.7, 2.4), rng.uniform(0.8, 1.5)])
        try:
            FiveBarDynamics(GEOM, PAR).mass_matrix(q)
        except ValueError:
            continue
        out.append(q)
    return out


@pytest.mark.parametrize("gravity", [(0.0, 0.0), (0.0, -9.81)])
def test_mass_matrix_spd(gravity):
    dyn = FiveBarDynamics(GEOM, PAR, gravity=gravity)
    for q in _samples(80, 1):
        M = dyn.mass_matrix(q)
        assert np.allclose(M, M.T, atol=1e-12)
        assert np.linalg.eigvalsh(M).min() > 0


def test_passivity_skew():
    dyn = FiveBarDynamics(GEOM, PAR)
    rng = np.random.default_rng(2)
    for q in _samples(20, 3):
        qd = rng.normal(0, 0.7, 2)
        h = 1e-6
        Mdot = (dyn.mass_matrix(q + h*qd) - dyn.mass_matrix(q - h*qd)) / (2*h)
        S = Mdot - 2*dyn.coriolis(q, qd)
        assert np.abs(S + S.T).max() < 1e-7      # skew-symmetric


def test_inverse_forward_consistency():
    rng = np.random.default_rng(4)
    err = 0.0
    for gravity in [(0.0, 0.0), (0.0, -9.81)]:
        dyn = FiveBarDynamics(GEOM, PAR, gravity=gravity)
        for q in _samples(40, 5):
            qd = rng.normal(0, 0.6, 2); qdd = rng.normal(0, 2, 2)
            w = rng.normal(0, 5, 3)
            tau = dyn.inverse_dynamics(q, qd, qdd, wrench=w)
            qdd2 = dyn.forward_dynamics(q, qd, tau, wrench=w)
            err = max(err, np.linalg.norm(qdd2 - qdd))
    assert err < 1e-10


def test_gravity_matches_potential_gradient():
    dyn = FiveBarDynamics(GEOM, PAR, gravity=(0.0, -9.81))
    h = 1e-6
    for q in _samples(20, 6):
        gnum = np.array([
            (dyn.potential_energy(q + np.array([h, 0])) -
             dyn.potential_energy(q - np.array([h, 0]))) / (2*h),
            (dyn.potential_energy(q + np.array([0, h])) -
             dyn.potential_energy(q - np.array([0, h]))) / (2*h)])
        assert np.allclose(dyn.gravity_torque(q), gnum, atol=1e-5)


def test_horizontal_gravity_zero():
    dyn = FiveBarDynamics(GEOM, PAR, gravity=(0.0, 0.0))
    for q in _samples(10, 7):
        assert np.allclose(dyn.gravity_torque(q), 0.0)
        assert dyn.potential_energy(q) == 0.0


def test_energy_conservation_coast():
    # horizontal free coast: kinetic energy must be conserved (RK4)
    dyn = FiveBarDynamics(GEOM, PAR, gravity=(0.0, 0.0))
    q = np.array([1.95, 1.19]); qd = np.array([1.2, 1.2])
    st = np.concatenate([q, qd]); dt = 2e-4
    E0 = dyn.kinetic_energy(q, qd)

    def deriv(s):
        return np.concatenate([s[2:], dyn.forward_dynamics(s[:2], s[2:],
                                                            np.zeros(2))])
    for _ in range(2000):
        k1 = deriv(st); k2 = deriv(st+dt/2*k1)
        k3 = deriv(st+dt/2*k2); k4 = deriv(st+dt*k3)
        st = st + dt/6*(k1+2*k2+2*k3+k4)
    E1 = dyn.kinetic_energy(st[:2], st[2:])
    assert abs(E1 - E0) / E0 < 1e-8


def test_end_effector_jacobian_and_moment():
    dyn = FiveBarDynamics(GEOM, PAR, tool_leg=0)
    q = np.array([2.0, 1.1])
    Je = dyn.end_effector_jacobian(q)
    assert Je.shape == (3, 2)
    # a pure moment maps through the tool-link passive rate h
    F = np.array([3.0, -2.0]); Mz = 0.7
    tau_full = dyn._ext_genforce(q, np.array([F[0], F[1], Mz]))
    tau_force = dyn._ext_genforce(q, F)
    assert np.allclose(tau_full - tau_force, Mz * Je[2], atol=1e-12)


def test_urdf_loader_matches_uniform():
    geom, par = load_urdf(URDF)
    assert np.isclose(geom.l1, 0.20) and np.isclose(geom.l3, 0.25)
    assert np.isclose(geom.d, 0.20)
    assert np.isclose(par.apex_mass, 0.10)
    # dynamics from URDF must equal dynamics from the uniform helper
    dyn_urdf, _ = build_dynamics(URDF, gravity=(0.0, -9.81))
    dyn_ref = FiveBarDynamics(GEOM, PAR, gravity=(0.0, -9.81))
    for q in _samples(20, 8):
        assert np.allclose(dyn_urdf.mass_matrix(q), dyn_ref.mass_matrix(q),
                           atol=1e-12)
        assert np.allclose(dyn_urdf.gravity_torque(q),
                           dyn_ref.gravity_torque(q), atol=1e-10)
