# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
simrobot.py -- a simulated robot for testing without hardware.

SimulatedRobot implements the RobotIO interface, so any code that drives a
real robot can be pointed at it instead.  It can mimic two realistic effects:
finite encoder resolution (counts_per_rev) and a finite servo speed limit
(qdot_max).  It also exposes the true end-effector position for checking
estimator accuracy in tests.
"""

import numpy as np
from robotio import RobotIO
from fkine import fk


class SimulatedRobot(RobotIO):
    """A two-joint robot living in software.

    Parameters
    ----------
    geom : FiveBarGeometry
    q0 : initial joint angles (radians).
    counts_per_rev : if given, read_q returns angles quantized to this
        encoder resolution; if None, encoders are perfect.
    qdot_max : if given, the joints move toward the commanded target no faster
        than this (rad/s); if None, the target is reached within one step.
    """

    def __init__(self, geom, q0, counts_per_rev=None, qdot_max=None):
        self.geom = geom
        self.q = np.asarray(q0, float).copy()
        self.q_target = self.q.copy()
        self.qdot = np.zeros(2)
        self.cpr = counts_per_rev
        self.qdot_max = qdot_max

    # --- RobotIO contract ------------------------------------------------
    def read_q(self):
        """Measured joint angles, quantized to the encoder grid if one is set."""
        if self.cpr is None:
            return self.q.copy()
        res = 2.0 * np.pi / self.cpr
        return np.round(self.q / res) * res

    def read_dq(self):
        """Measured joint rates from the last step."""
        return self.qdot.copy()

    def write_q(self, q_cmd):
        """Set the target the joints will move toward on the next step."""
        self.q_target = np.asarray(q_cmd, float).copy()

    # --- plant update ----------------------------------------------------
    def step(self, dt):
        """Advance the simulated joints by one time step dt (seconds)."""
        error = self.q_target - self.q
        if self.qdot_max is None:
            self.qdot = error / dt if dt > 0 else np.zeros(2)
            self.q = self.q_target.copy()
        else:
            step = np.clip(error, -self.qdot_max * dt, self.qdot_max * dt)
            self.qdot = step / dt if dt > 0 else np.zeros(2)
            self.q = self.q + step

    # --- ground truth for tests -----------------------------------------
    def true_X(self, sigma=+1):
        """The exact end-effector position of the current joints (no
        quantization), for comparison against an estimate."""
        return fk(self.geom, self.q, sigma)
