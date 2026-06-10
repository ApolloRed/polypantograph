import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))
import numpy as np
from robotgeometry import FiveBarGeometry
from ikine import ik
from jacobians import inv_condition, assembly_mode
from virtualwall import BoxWall, CertifiedRegion, ChainedProjection

G = FiveBarGeometry.symmetric(0.20, 0.25, 0.20)


def test_box_wall():
    w = BoxWall(-0.12, 0.06, 0.20, 0.40)
    assert np.allclose(w.project([0.10, 0.30]), [0.06, 0.30])
    assert w.contains([0.0, 0.30]) and not w.contains([0.10, 0.30])
    assert np.allclose(w.tangentialize([0.06, 0.30], [1.0, 0.5]), [0.0, 0.5])
    assert np.allclose(w.tangentialize([0.0, 0.30], [1.0, 0.5]), [1.0, 0.5])


def test_certified_region_invariants_and_projection():
    reg = CertifiedRegion.build(G, kappa0=0.4, reach_margin=0.005,
                                bounds=(-0.30, 0.30, 0.0, 0.45), cell=0.005)
    assert reg.contains([0.0, 0.30])
    rng = np.random.default_rng(5)
    for i in rng.integers(0, len(reg.points), 50):
        X = reg.points[i]
        q = ik(G, X, (+1, -1))
        assert inv_condition(G, q, X) >= reg.kappa0 - 1e-9
        assert assembly_mode(G, q, X) == +1
    assert reg.contains(reg.project([0.0, 0.02]))
    chained = ChainedProjection(reg, BoxWall(-0.12, 0.06, 0.20, 0.40))
    assert chained.project([0.10, 0.30])[0] <= 0.06 + 1e-12
