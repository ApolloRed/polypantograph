# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
trajectorygen.py -- simple end-effector trajectory generators.

Each generator returns a function f(t) -> (X_desired, Xdot_desired): the
desired position and velocity at time t.  These are only a convenience for
pre-planned moves; in admittance/impedance or teleoperation the command is
produced live instead (see examples/hardware_IK.py).
"""

import numpy as np


def circle(center, radius, period, t0=0.0, phase=0.0):
    """A counter-clockwise circle of given radius and period (seconds)."""
    center = np.asarray(center, float)
    omega = 2.0 * np.pi / period

    def f(t):
        theta = omega * (t - t0) + phase
        X = center + radius * np.array([np.cos(theta), np.sin(theta)])
        Xdot = radius * omega * np.array([-np.sin(theta), np.cos(theta)])
        return X, Xdot
    return f


def line(p0, p1, duration, smooth=True):
    """A straight move from p0 to p1 over duration seconds.  With
    smooth=True the speed follows a minimum-jerk profile (zero velocity at
    both ends); with smooth=False it is constant."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)

    def f(t):
        s = np.clip(t / duration, 0.0, 1.0)
        if smooth:
            tau = s
            s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
            sdot = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / duration
        else:
            sdot = 1.0 / duration if 0.0 <= t <= duration else 0.0
        return p0 + s * (p1 - p0), sdot * (p1 - p0)
    return f


def hold(point):
    """Stay at a fixed point (zero velocity)."""
    point = np.asarray(point, float)

    def f(t):
        return point.copy(), np.zeros(2)
    return f


def trajconcat(*segments):
    """Concatenate trajectory segments into one trajectory.

    Each argument is a (duration, trajectory) pair.  The returned function
    plays them back to back, shifting time so each segment starts at zero.
    """
    durations = [d for d, _ in segments]
    edges = np.concatenate([[0.0], np.cumsum(durations)])

    def f(t):
        t = min(max(t, 0.0), edges[-1] - 1e-12)
        i = int(np.searchsorted(edges, t, side="right") - 1)
        return segments[i][1](t - edges[i])
    return f
