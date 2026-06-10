# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
robotgeometry.py -- the mechanism's fixed dimensions and a couple of basic
geometric helpers that almost every other file needs.

A planar five-bar (pantograph) has two driven "proximal" links and two
passive "distal" links meeting at the end-effector, plus the fixed base.
We allow four DIFFERENT link lengths (the general case); the common
symmetric robot is just the special case l1=l2 and l3=l4, available through
the .symmetric() shortcut.

Frame: the origin sits halfway between the two motors, so motor 1 is at
(-d/2, 0) and motor 2 at (+d/2, 0).  All angles are in radians, measured
counter-clockwise from the +x axis; all lengths are in meters.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FiveBarGeometry:
    """Immutable container for the five link lengths.

    Attributes
    ----------
    l1, l2 : float
        Proximal (motor-driven) link lengths of leg 1 and leg 2.
    l3, l4 : float
        Distal (passive) link lengths of leg 1 and leg 2.
    d : float
        Distance between the two motor shafts (the fixed "base" link, l5).
    """
    l1: float
    l2: float
    l3: float
    l4: float
    d: float

    def __post_init__(self):
        # Reject nonsensical dimensions early -- a zero or negative length
        # would silently produce wrong kinematics later on.
        for name in ("l1", "l2", "l3", "l4", "d"):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive length")

    @classmethod
    def symmetric(cls, a, b, d):
        """Build the common symmetric robot: both proximal links = a, both
        distal links = b, base = d.  This is the common symmetric robot."""
        return cls(a, a, b, b, d)

    @property
    def is_symmetric(self):
        """True when the two legs are identical (l1==l2 and l3==l4)."""
        return np.isclose(self.l1, self.l2) and np.isclose(self.l3, self.l4)

    @property
    def bases(self):
        """The two motor positions as a 2x2 array; row 0 is O1, row 1 is O2."""
        return np.array([[-self.d / 2.0, 0.0], [self.d / 2.0, 0.0]])

    @property
    def proximal(self):
        """Proximal link lengths as the array [l1, l2]."""
        return np.array([self.l1, self.l2])

    @property
    def distal(self):
        """Distal link lengths as the array [l3, l4]."""
        return np.array([self.l3, self.l4])

    def reach_annulus(self, leg):
        """Smallest and largest distance the given leg (0 or 1) can place the
        end-effector from its motor.  A leg of links (lp, ld) can only reach
        points whose distance r from the motor satisfies
        |lp - ld| <= r <= lp + ld.  Returns (r_min, r_max)."""
        lp = self.proximal[leg]
        ld = self.distal[leg]
        return abs(lp - ld), lp + ld


def wrap(angle):
    """Wrap an angle (or array of angles) into the interval (-pi, pi].

    Joint angles coming out of inverse kinematics can land outside that
    range; wrapping keeps comparisons (e.g. "did the joint jump?") sensible.
    """
    return -((-np.asarray(angle, float) + np.pi) % (2.0 * np.pi) - np.pi)


def elbows(geom, q):
    """Positions of the two elbow (passive) joints for motor angles q.

    Each elbow E_i sits at the end of its proximal link:
        E_i = O_i + l_i * (cos q_i, sin q_i).
    This is the most-used primitive in the whole library, so it lives here
    in the geometry module.

    Parameters
    ----------
    geom : FiveBarGeometry
    q : array-like, shape (2,)
        Motor angles (q1, q2) in radians.

    Returns
    -------
    (2, 2) array; row 0 is elbow E1, row 1 is elbow E2.
    """
    q = np.asarray(q, float)
    direction = np.stack([np.cos(q), np.sin(q)], axis=1)
    return geom.bases + geom.proximal[:, None] * direction
