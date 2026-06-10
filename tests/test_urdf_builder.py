import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))
import numpy as np
from robotgeometry import FiveBarGeometry
from dynamics import DynamicParameters
from urdf_builder import build_urdf_string, write_urdf, write_urdf_symmetric
from urdf_loader import load_urdf


def test_builder_loader_roundtrip(tmp_path):
    """A URDF written by the builder loads back to the same parameters."""
    geom = FiveBarGeometry(l1=0.19, l2=0.21, l3=0.24, l4=0.26, d=0.18)
    par = DynamicParameters(proximal_mass=[0.50, 0.55],
                            proximal_com=[0.09, 0.10],
                            proximal_inertia=[0.0015, 0.0016],
                            distal_mass=[0.30, 0.32], distal_com=[0.12, 0.13],
                            distal_inertia=[0.0014, 0.0015], apex_mass=0.08)
    path = str(tmp_path / "custom.urdf")
    write_urdf(path, geom, par, name="custom_arm")
    g2, p2 = load_urdf(path)
    assert np.allclose([g2.l1, g2.l2, g2.l3, g2.l4, g2.d],
                       [0.19, 0.21, 0.24, 0.26, 0.18])
    assert np.allclose(p2.proximal_mass, par.proximal_mass)
    assert np.allclose(p2.proximal_com, par.proximal_com)
    assert np.allclose(p2.proximal_inertia, par.proximal_inertia)
    assert np.allclose(p2.distal_mass, par.distal_mass)
    assert np.allclose(p2.distal_inertia, par.distal_inertia)
    assert np.isclose(p2.apex_mass, par.apex_mass)
    assert 'name="custom_arm"' in build_urdf_string(geom, par, "custom_arm")


def test_symmetric_helper(tmp_path):
    path = str(tmp_path / "sym.urdf")
    write_urdf_symmetric(path, "sym", a=0.20, b=0.25, d=0.20,
                         proximal_mass=0.5, distal_mass=0.3, apex_mass=0.10)
    geom, par = load_urdf(path)
    assert np.isclose(geom.l1, 0.20) and np.isclose(geom.l3, 0.25)
    assert np.isclose(geom.d, 0.20)
    assert np.isclose(par.apex_mass, 0.10)
    assert np.allclose(par.proximal_inertia, 0.5 * 0.20 ** 2 / 12.0)
