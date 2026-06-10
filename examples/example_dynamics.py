# polypantograph example -- Author: Abolfazl Mohebbi, PhD., Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
example_dynamics.py -- model-based (computed-torque) tracking with the
five-bar dynamic model, all parameters loaded from a URDF.

The robot is placed in the vertical mounting (gravity active).  The
end-effector is commanded around a circle; at every sample the inverse
dynamics provides the feedforward torque of a computed-torque law, and the
forward dynamics advances a simulated plant.  This is the model-based inner
loop an interaction/rehabilitation controller would sit on top of.

To move this to a real robot:
  * read the measured joint angles from your RobotIO subclass instead of the
    simulated plant (see example_IK.py / example_FK.py),
  * send the computed torque tau to your current-controlled drives,
  * keep the same inverse_dynamics() call for the feedforward.
"""

import numpy as np
from ikine import ik
from urdf_loader import build_dynamics

HERE = os.path.dirname(__file__)
OMEGA, RATE, T = (+1, -1), 1000.0, 3.0
dt = 1.0 / RATE

# build the dynamics straight from the URDF (vertical mounting)
# dyn, geom = build_dynamics(os.path.join(HERE, "fivebar.urdf"), gravity=(0.0, -9.81))
dyn, geom = build_dynamics(os.path.join(HERE, "general_arm.urdf"), gravity=(0.0, -9.81))


# circular task reference -> joint reference via inverse kinematics
t = np.arange(int(T * RATE)) * dt
w = 2 * np.pi / T
Xref = np.stack([0.06 * np.cos(w * t), 0.30 + 0.06 * np.sin(w * t)], axis=1)
Qref = np.array([ik(geom, X, OMEGA) for X in Xref])
Qd_ref = np.gradient(Qref, dt, axis=0)
Qdd_ref = np.gradient(Qd_ref, dt, axis=0)

# computed-torque control of a simulated plant
Kp, Kd = 400.0, 40.0
q = Qref[0] + np.array([0.05, -0.04])           # deliberate initial error
qd = np.zeros(2)
err, peak_tau = [], 0.0
for k in range(len(t)):
    e = Qref[k] - q
    aq = Qdd_ref[k] + Kd * (Qd_ref[k] - qd) + Kp * e
    tau = dyn.inverse_dynamics(q, qd, aq)        # feedforward + feedback
    q, qd = dyn.step(q, qd, tau, dt)             # advance the plant (FD)
    err.append(np.linalg.norm(e)); peak_tau = max(peak_tau, np.abs(tau).max())
err = np.array(err)
print("computed-torque tracking (vertical mounting):")
print(f"  initial error  {err[0]*1e3:6.1f} mrad")
print(f"  settled error  {err[-200:].max()*1e3:8.3e} mrad")
print(f"  peak |tau|     {peak_tau:6.3f} N.m")

try:
    import matplotlib.pyplot as plt
    import plotting
    plotting.setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.4))
    ax[0].plot(Xref[:, 0], Xref[:, 1], "k--", lw=0.8, label="reference")
    ax[0].set_aspect("equal"); ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("task circle", fontsize=8)
    ax[1].semilogy(t, np.maximum(err, 1e-9), lw=0.9)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("joint error [rad]")
    ax[1].set_title("computed-torque convergence", fontsize=8)
    fig.tight_layout(); fig.savefig("example_dynamics.pdf")
    print("  plot saved: example_dynamics.pdf")
except ImportError:
    pass
