# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
robotio.py -- interface to the hardware API.

This file does NOT talk to any specific motor or bus.  It only defines the
small contract the rest of the library expects from "the robot": read the two
joint angles, and command the two joint angles, both in radians.  You write a
short subclass of RobotIO that calls your own servo API, and everything else
in the library works against it unchanged.  The simulated robot in simrobot.py
implements the very same contract for testing.

A real drive typically reports joint angles directly (already converted from
its encoder), so no count-to-radian conversion lives here -- your subclass
just returns the angles its API gives you.
"""

from abc import ABC, abstractmethod


class RobotIO(ABC):
    """Minimal two-joint robot interface (all values in radians).

    Subclass this and implement read_q and write_q for your hardware.  If your
    drives also report speed, override read_dq to return it; otherwise the
    default returns None and callers fall back to finite differencing.
    """

    @abstractmethod
    def read_q(self):
        """Return the measured joint angles as a length-2 array (radians)."""

    @abstractmethod
    def write_q(self, q_cmd):
        """Command target joint angles (length-2, radians)."""

    def read_dq(self):
        """Measured joint rates (length-2, rad/s), or None if unavailable."""
        return None
