# polypantograph example -- Author: Abolfazl Mohebbi, PhD., Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
example_FK.py -- per-sample forward kinematics for LIVE pose estimation.

Joint motion is produced one sample at a time (here an operator-style jog; on
hardware, whatever your motors do), and each sample the estimator turns the
measured, quantized encoder readings into the end-effector pose and velocity,
plus live monitors.  Those monitors drive a safety reflex: when the measured
conditioning approaches the floor, the jog is overridden by a retreat toward
the homing posture until it recovers (a plain freeze would deadlock, parked
below the floor forever).

To move this to real hardware:
  * replace SimulatedRobot with your own RobotIO subclass,
  * replace jog() with your command source (or none -- estimation works the
    same if the joints are moved by hand),
  * call estimator.update() from your real-time scheduler.
"""

import numpy as np
from robotgeometry import FiveBarGeometry, wrap
from ikine import ik
from simrobot import SimulatedRobot
from fkine import FKNewtonEstimator
from jacobians import jacobian, singular_values
from urdf_loader import build_dynamics

# --- setup ---------------------------------------------------------------

# --- general robot ---
# HERE = os.path.dirname(__file__)
# dyn, geom = build_dynamics(os.path.join(HERE, "general_arm.urdf"), gravity=(0.0, -9.81))

# --- symmetric uniform robot ---
geom = FiveBarGeometry.symmetric(a=0.20, b=0.25, d=0.20)

SIGMA, RATE = +1, 500.0
DT = 1.0 / RATE
CPR = 2 ** 16
ENC_RES = 2 * np.pi / CPR      # encoder resolution at the joint (rad)
KAPPA_FLOOR = 0.35

robot = SimulatedRobot(geom, ik(geom, [0.0, 0.30], (+1, -1)),  # <-- your RobotIO
                       counts_per_rev=CPR)
estimator = FKNewtonEstimator(geom, sigma0=SIGMA)
estimator.reset(robot.read_q())


# -------------------------------------------------------------------------
# USING A REAL ROBOT
# -------------------------------------------------------------------------
# Replace the SimulatedRobot above with a small RobotIO subclass that talks to
# your own motor API.  The estimator only needs measured joint angles, which a
# real drive already provides in radians.
#
#   from robotio import RobotIO
#   import numpy as np
#
#   class MyRobot(RobotIO):
#       def __init__(self, api):
#           self.api = api
#       def read_q(self):
#           return np.array(self.api.get_joint_angles())   # radians
#       def write_q(self, q_cmd):
#           self.api.set_joint_targets(float(q_cmd[0]), float(q_cmd[1]))
#
#   robot = MyRobot(my_api)
#   estimator = FKNewtonEstimator(geom, sigma0=SIGMA)
#   estimator.reset(robot.read_q())
#   # then each control sample:
#   #     q_meas = robot.read_q()
#   #     est = estimator.update(q_meas, qdot_meas)   # qdot_meas optional
#   #     ... use est.X, est.Xdot, est.inv_cond, est.det_A ...
#   # ENC_RES above is just 2*pi / counts_per_rev for your encoder, used only
#   # to draw the resolution bound; it is not needed by the estimator itself.
# -------------------------------------------------------------------------


def jog(t):
    """Operator jog rates (rad/s).  The first phase spreads the arms, driving
    the apex down toward the interior Type-2 locus to exercise the reflex."""
    if t < 1.2:
        return np.array([+0.8, -0.8])
    if t < 2.0:
        return np.array([-0.5, +0.6])
    return np.array([0.5 * np.sin(2 * np.pi * (t - 2.0)), -0.2])


# --- the real-time loop (one sample at a time) ---------------------------
q_home = robot.read_q()
q_cmd = q_home.copy()
q_meas_prev = robot.read_q()
hist = {"t": [], "X": [], "ic": [], "veto": [], "err": [], "bound": []}
for k in range(int(3.0 * RATE)):
    t = k * DT
    q_meas = robot.read_q()                              # 1. measured joints
    X_true_now = robot.true_X(SIGMA)                     # truth AT this read
    qdot_meas = wrap(q_meas - q_meas_prev) / DT
    q_meas_prev = q_meas
    est = estimator.update(q_meas, qdot_meas)            # 2. one FK tick
    vetoed = est.inv_cond < KAPPA_FLOOR                  # 3. online decision
    v = 2.0 * wrap(q_home - q_cmd) if vetoed else jog(t)  # retreat or jog
    v = np.clip(v, -1.0, 1.0)
    q_cmd = q_cmd + DT * v                                # 4. act
    robot.write_q(q_cmd)
    robot.step(DT)
    bound = singular_values(jacobian(geom, q_meas, est.X))[0] * ENC_RES
    hist["t"].append(t); hist["X"].append(est.X)
    hist["ic"].append(est.inv_cond); hist["veto"].append(vetoed)
    hist["err"].append(np.linalg.norm(est.X - X_true_now))
    hist["bound"].append(bound)

X = np.array(hist["X"]); ic = np.array(hist["ic"])
err = np.array(hist["err"]); veto = np.array(hist["veto"])
bound = np.array(hist["bound"])
print("samples=%d  FK error: mean %.1f um, max %.1f um; within the live "
      "||J|| dq_enc bound at %.1f%% of samples"
      % (len(X), err.mean() * 1e6, err.max() * 1e6,
         (err <= bound + 1e-9).mean() * 100))
print("min measured 1/kappa = %.3f; retreat reflex active on %d samples "
      "(%.1f%%), then recovered"
      % (ic.min(), int(veto.sum()), 100 * veto.mean()))

try:
    import matplotlib.pyplot as plt
    import plotting
    plotting.setup_style()
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.6))
    ax[0].plot(X[:, 0], X[:, 1], lw=0.9); ax[0].set_aspect("equal")
    ax[0].set_title("live FK estimate of the path", fontsize=8)
    ax[1].plot(hist["t"], ic, lw=0.9)
    ax[1].axhline(KAPPA_FLOOR, color="tab:red", lw=0.8, ls=":")
    ax[1].fill_between(hist["t"], 0, 1, where=veto, alpha=0.15,
                       color="tab:red")
    ax[1].set_ylim(0, 1); ax[1].set_title(r"live $1/\kappa$ + veto",
                                          fontsize=8)
    ax[2].plot(hist["t"], err * 1e6, lw=0.7)
    ax[2].set_title(r"estimation error [$\mu$m]", fontsize=8)
    fig.tight_layout(); fig.savefig("example_FK.pdf")
    print("plot saved: example_FK.pdf")
except ImportError:
    pass
