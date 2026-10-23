# polypantograph -- kinematics and dynamics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
urdf_loader.py -- read the five-bar's geometry and inertial parameters from a
URDF file.

URDF describes strictly tree-structured robots and cannot encode a closed
loop, so the five-bar is given as a *spanning tree*: two serial 2R legs hanging
off a common base, each ending in a fixed "tip" frame at the end-effector.  The
loop closure (the two tips coincide at the apex) is not written in the file --
it is supplied by the five-bar model itself.  The URDF therefore acts as a
parameter container: link lengths come from the joint origins, and masses,
centre-of-mass offsets, and inertias come from the <inertial> blocks.

Expected structure (names are free; the tree is walked generically):

    base
      |-- motor joint (revolute/continuous, axis z) --> proximal link
      |     |-- elbow joint (revolute/continuous)    --> distal link
      |     |     |-- tip joint (fixed)              --> apex/tool link
      |-- (second leg, identically)

Conventions: each link's local frame sits at its inboard joint with x along
the link, so a link length is the x-offset of the next joint's <origin>, and a
centre-of-mass offset is the x of the <inertial><origin>.  Motion is planar
about z, so the inertia used is izz.  The two legs are ordered by motor x
(leg 1 is the one at negative x).  The apex/payload mass is the sum of the tip
links' masses (the two tips are the same physical point).

Only the Python standard library is used (xml.etree); no external URDF parser.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np

from .robotgeometry import FiveBarGeometry
from .dynamics import DynamicParameters, FiveBarDynamics


def _xyz(elem, default=(0.0, 0.0, 0.0)):
    """Read an xyz triple from an <origin> child of elem (or default)."""
    if elem is None:
        return np.array(default, float)
    origin = elem.find("origin")
    if origin is None or "xyz" not in origin.attrib:
        return np.array(default, float)
    return np.array([float(v) for v in origin.attrib["xyz"].split()], float)


def _inertial(link):
    """Return (mass, com_x, izz) from a <link>'s <inertial> block; zeros if
    absent."""
    inertial = link.find("inertial")
    if inertial is None:
        return 0.0, 0.0, 0.0
    mass_el = inertial.find("mass")
    mass = float(mass_el.attrib.get("value", 0.0)) if mass_el is not None else 0.0
    com_x = _xyz(inertial)[0]
    izz = 0.0
    inertia = inertial.find("inertia")
    if inertia is not None:
        izz = float(inertia.attrib.get("izz", 0.0))
    return mass, com_x, izz


def load_urdf(path):
    """Parse a five-bar URDF and return (geometry, parameters).

    Returns
    -------
    geom : FiveBarGeometry
    params : DynamicParameters

    Raises
    ------
    ValueError if the file does not have the expected two-leg tree.
    """
    root = ET.parse(path).getroot()
    links = {ln.attrib["name"]: ln for ln in root.findall("link")}
    joints = root.findall("joint")

    def parent_of(j):
        return j.find("parent").attrib["link"]

    def child_of(j):
        return j.find("child").attrib["link"]

    # the base is the only link that is never a child
    children = {child_of(j) for j in joints}
    bases = [name for name in links if name not in children]
    if len(bases) != 1:
        raise ValueError("expected exactly one base link, found %d" % len(bases))
    base = bases[0]
    by_parent = {}
    for j in joints:
        by_parent.setdefault(parent_of(j), []).append(j)

    # the two actuated joints hang off the base
    motors = [j for j in by_parent.get(base, [])
              if j.attrib.get("type") in ("revolute", "continuous")]
    if len(motors) != 2:
        raise ValueError("expected two actuated joints on the base, found %d"
                         % len(motors))
    motors.sort(key=lambda j: _xyz(j)[0])      # leg 1 at negative x

    O = np.zeros((2, 2))
    lp = np.zeros(2); ld = np.zeros(2)
    pm = np.zeros(2); pc = np.zeros(2); pI = np.zeros(2)
    dm = np.zeros(2); dcom = np.zeros(2); dI = np.zeros(2)
    apex_mass = 0.0
    for i, motor in enumerate(motors):
        O[i] = _xyz(motor)[:2]
        proximal = child_of(motor)
        elbow = by_parent[proximal][0]         # proximal -> distal
        lp[i] = _xyz(elbow)[0]
        pm[i], pc[i], pI[i] = _inertial(links[proximal])
        distal = child_of(elbow)
        tip = by_parent[distal][0]             # distal -> apex
        ld[i] = _xyz(tip)[0]
        dm[i], dcom[i], dI[i] = _inertial(links[distal])
        apex_mass += _inertial(links[child_of(tip)])[0]

    d = abs(O[1, 0] - O[0, 0])
    geom = FiveBarGeometry(l1=lp[0], l2=lp[1], l3=ld[0], l4=ld[1], d=d)
    params = DynamicParameters(proximal_mass=pm, proximal_com=pc,
                               proximal_inertia=pI, distal_mass=dm,
                               distal_com=dcom, distal_inertia=dI,
                               apex_mass=apex_mass)
    return geom, params


def build_dynamics(path, gravity=(0.0, 0.0), sigma=+1, tool_leg=0):
    """Convenience: load a URDF and return a ready FiveBarDynamics, along with
    the geometry.  Returns (dynamics, geometry)."""
    geom, params = load_urdf(path)
    return FiveBarDynamics(geom, params, gravity=gravity, sigma=sigma,
                           tool_leg=tool_leg), geom
