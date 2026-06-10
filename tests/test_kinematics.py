import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))
import numpy as np
import pytest
from robotgeometry import FiveBarGeometry, elbows, wrap
from fkine import fk, constraint
from ikine import ik, ik_branches, working_modes
from jacobians import det_A

GEN = FiveBarGeometry(0.20, 0.24, 0.26, 0.32, 0.20)
SYM = FiveBarGeometry.symmetric(0.20, 0.25, 0.20)


@pytest.mark.parametrize("geom", [GEN, SYM])
def test_fk_satisfies_loop_closure(geom):
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(0.3, 2.8, 2)
        for sigma in (+1, -1):
            try:
                X = fk(geom, q, sigma)
            except ValueError:
                continue
            b = X - elbows(geom, q)
            assert np.allclose(np.linalg.norm(b, axis=1), geom.distal,
                               atol=1e-12)
            assert np.allclose(constraint(geom, q, X), 0.0, atol=1e-12)


@pytest.mark.parametrize("geom", [GEN, SYM])
def test_round_trip_ik_fk(geom):
    rng = np.random.default_rng(1)
    n = 0
    for _ in range(300):
        q = rng.uniform(0.3, 2.8, 2)
        try:
            X = fk(geom, q, +1)
        except ValueError:
            continue
        om = working_modes(geom, q, X)
        assert np.allclose(wrap(ik(geom, X, om) - q), 0.0, atol=1e-10)
        n += 1
    assert n > 100


def test_detA_equals_sigma_h_D_general():
    rng = np.random.default_rng(2)
    for _ in range(200):
        q = rng.uniform(0.3, 2.8, 2)
        E = elbows(GEN, q)
        P = E[1] - E[0]
        D = np.linalg.norm(P)
        e = (D * D + GEN.l3 ** 2 - GEN.l4 ** 2) / (2 * D)
        h2 = GEN.l3 ** 2 - e * e
        if h2 <= 1e-9:
            continue
        h = np.sqrt(h2)
        for sigma in (+1, -1):
            X = fk(GEN, q, sigma)
            assert np.isclose(det_A(GEN, q, X), sigma * h * D, atol=1e-12)


def test_ik_branches_and_unreachable():
    X = np.array([0.0, 0.30])
    br = ik_branches(GEN, X)
    assert len(br) == 4
    for q in br.values():
        assert np.allclose(constraint(GEN, q, X), 0.0, atol=1e-10)
    with pytest.raises(ValueError):
        ik(GEN, [2.0, 2.0])


def test_symmetric_foot_is_midpoint():
    q = np.array([1.0, 2.0])
    E = elbows(SYM, q)
    X = fk(SYM, q, +1)
    P = E[1] - E[0]
    e = (X - E[0]) @ (P / np.linalg.norm(P))
    assert np.isclose(e, np.linalg.norm(P) / 2, atol=1e-12)
