# polypantograph example -- Author: Abolfazl Mohebbi, PhD., Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
example_IK.py -- per-sample inverse kinematics in an ADMITTANCE loop.

The end-effector command does not exist before the loop runs: at every sample
a measured force is turned into a displacement by a virtual mass-damper, and
that becomes this sample's IK target.  The pipeline is therefore driven one
sample at a time with TrackingController.step().

To move this to real hardware:
  * replace SimulatedRobot with your own RobotIO subclass,
  * replace read_force() with your force/torque sensor read,
  * call ctrl.step() from your real-time scheduler instead of this for-loop.
"""

import numpy as np
from robotgeometry import FiveBarGeometry
from ikine import ik
from simrobot import SimulatedRobot
from virtualwall import BoxWall
from ikine import PerLegNewtonSolver
from realtime_looper import TrackingController
from urdf_loader import build_dynamics

# --- setup ---------------------------------------------------------------

# --- general robot ---
HERE = os.path.dirname(__file__)
dyn, geom = build_dynamics(os.path.join(HERE, "general_arm.urdf"), gravity=(0.0, -9.81))

# --- symmetric uniform robot ---
# geom = FiveBarGeometry.symmetric(a=0.20, b=0.25, d=0.20)

OMEGA, SIGMA, RATE = (+1, -1), +1, 500.0
DT = 1.0 / RATE
X_HOME = np.array([0.0, 0.30])

robot = SimulatedRobot(geom, ik(geom, X_HOME, OMEGA),     # <-- your RobotIO
                       counts_per_rev=2 ** 16)
wall = BoxWall(-0.10, 0.10, 0.22, 0.38)                   # safety keep-in box
ctrl = TrackingController(robot, geom, PerLegNewtonSolver(geom),
                          projection=wall, rate_hz=RATE, sigma0=SIGMA)

# -------------------------------------------------------------------------
# USING A REAL ROBOT
# -------------------------------------------------------------------------
# Replace the SimulatedRobot above with a small RobotIO subclass that talks to
# your own motor API.  Everything else in this script stays exactly the same.
#
#   from robotio import RobotIO
#   import numpy as np
#
#   class MyRobot(RobotIO):
#       def __init__(self, api):
#           self.api = api                 # your motor/driver handle
#       def read_q(self):
#           # return the two joint angles in radians as a length-2 array
#           return np.array(self.api.get_joint_angles())
#       def write_q(self, q_cmd):
#           # command the two joint angles in radians
#           self.api.set_joint_targets(float(q_cmd[0]), float(q_cmd[1]))
#       def read_dq(self):                 # optional, if your drives report speed
#           return np.array(self.api.get_joint_speeds())
#
#   robot = MyRobot(my_api)
#   ctrl = TrackingController(robot, geom, PerLegNewtonSolver(geom),
#                             projection=wall, rate_hz=RATE, sigma0=SIGMA)
#   # then call ctrl.step(X_d, V_des=v) from your real-time scheduler, with
#   # X_d/v produced by the admittance law each sample, just as below.
#   # (Drop robot.step(); a real robot advances on its own.)
# -------------------------------------------------------------------------

# virtual admittance: M a + D v = F, plus a soft spring back toward home
M_a, D_a, K_home = 2.0, 12.0, 25.0       # kg, N s/m, N/m


def read_force(t):
    """Stand-in for a force/torque sensor read."""
    F = np.zeros(2)
    if 0.5 <= t < 1.2:
        F += np.array([+3.0, 0.0])
    if 1.8 <= t < 2.5:
        F += np.array([0.0, -2.0])
    return F


# --- the real-time loop (one sample at a time) ---------------------------
X_d = X_HOME.copy()
v = np.zeros(2)
for k in range(int(3.5 * RATE)):
    t = k * DT
    F = read_force(t)                                    # 1. sense
    a = (F + K_home * (X_HOME - X_d) - D_a * v) / M_a     # 2. admittance law
    v = v + DT * a
    X_d = X_d + DT * v                                    # command this sample
    res = ctrl.step(X_d, V_des=v)                         # 3. one IK tick
    # res.X_exec / res.inv_cond / res.det_A are available right here.

log = ctrl.log()
print("admittance run:", log.summary())
print("wall respected: max x_cmd = %.6f (wall at %.2f)"
      % (log.X_cmd[:, 0].max(), wall.xmax))
print("peak displacement from home: %.1f mm"
      % (np.linalg.norm(log.X_cmd - X_HOME, axis=1).max() * 1e3))

try:
    import matplotlib.pyplot as plt
    import plotting
    plotting.setup_style()
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.6))
    ax[0].plot(log.X_des[:, 0], log.X_des[:, 1], "k--", lw=0.8,
               label="command")
    ax[0].plot(log.X_exec[:, 0], log.X_exec[:, 1], lw=0.9, label="executed")
    ax[0].axvline(wall.xmax, color="tab:red", lw=0.8)
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=6, frameon=False)
    ax[0].set_title("path (command made online)", fontsize=8)
    ax[1].plot(log.t, [read_force(t)[0] for t in log.t], label="$F_x$")
    ax[1].plot(log.t, [read_force(t)[1] for t in log.t], label="$F_y$")
    ax[1].legend(fontsize=6, frameon=False)
    ax[1].set_title("interaction force", fontsize=8)
    ax[2].plot(log.t, np.degrees(log.q_cmd))
    ax[2].set_title("joint commands", fontsize=8)
    fig.tight_layout(); fig.savefig("example_IK.pdf")
    print("plot saved: example_IK.pdf")
except ImportError:
    pass
