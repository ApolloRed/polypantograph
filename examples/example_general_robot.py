# polypantograph example -- Author: Abolfazl Mohebbi, PhD., Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
example_general_robot.py -- a GENERAL five-bar: asymmetric geometry (the two
legs have different link lengths) and non-uniform inertia (centre-of-mass
offsets are not L/2 and the link inertias are not the slender-rod value, and
they differ leg to leg).  This is the counterpart to the symmetric/uniform
examples.

It also shows the round trip with no URDF on hand: the parameters are written
to a URDF with urdf_builder (custom name), loaded back with urdf_loader, and
the dynamics are built from that.  A computed-torque controller then tracks a
circle to confirm the model drives a controller just like the symmetric case.
"""

import numpy as np
from robotgeometry import FiveBarGeometry
from dynamics import DynamicParameters
from urdf_builder import write_urdf
from urdf_loader import build_dynamics
from ikine import ik
from fkine import fk

HERE = os.path.dirname(__file__)
OMEGA, RATE, T = (+1, -1), 1000.0, 3.0
dt = 1.0 / RATE

# --- define a general robot by hand -------------------------------------
geom = FiveBarGeometry(l1=0.18, l2=0.22, l3=0.23, l4=0.27, d=0.19)
par = DynamicParameters(proximal_mass=[0.45, 0.55], proximal_com=[0.07, 0.12],
                        proximal_inertia=[0.0012, 0.0020],
                        distal_mass=[0.28, 0.34], distal_com=[0.10, 0.16],
                        distal_inertia=[0.0011, 0.0019], apex_mass=0.08)
print("general robot: proximal lengths %.2f / %.2f m, distal %.2f / %.2f m"
      % (geom.l1, geom.l2, geom.l3, geom.l4))
print("               non-uniform: com offsets and inertias differ per link")

# --- no URDF? write one, then load it back ------------------------------
urdf_path = os.path.join(HERE, "general_arm.urdf")
write_urdf(urdf_path, geom, par, name="general_arm")
dyn, geom = build_dynamics(urdf_path, gravity=(0.0, -9.81))    # vertical
print("wrote and reloaded:", os.path.basename(urdf_path))

# assembly mode consistent with omega = (+1, -1)
X0 = np.array([0.0, 0.30]); q_test = ik(geom, X0, OMEGA)
dyn.sigma = +1 if np.linalg.norm(fk(geom, q_test, +1) - X0) < 1e-9 else -1

# --- computed-torque tracking of a reachable circle ---------------------
t = np.arange(int(T * RATE)) * dt; w = 2 * np.pi / T
Xref = np.stack([0.05 * np.cos(w * t), 0.30 + 0.05 * np.sin(w * t)], axis=1)
Qref = np.array([ik(geom, X, OMEGA) for X in Xref])
Qd_ref = np.gradient(Qref, dt, axis=0); Qdd_ref = np.gradient(Qd_ref, dt, axis=0)
Kp, Kd = 400.0, 40.0
q = Qref[0] + np.array([0.04, -0.05]); qd = np.zeros(2)
err, peak, Xe = [], 0.0, []
for k in range(len(t)):
    e = Qref[k] - q
    aq = Qdd_ref[k] + Kd * (Qd_ref[k] - qd) + Kp * e
    tau = dyn.inverse_dynamics(q, qd, aq)
    q, qd = dyn.step(q, qd, tau, dt)
    err.append(np.linalg.norm(e)); peak = max(peak, np.abs(tau).max())
    Xe.append(fk(geom, q, dyn.sigma))
err = np.array(err); Xe = np.array(Xe)
print("computed-torque tracking (vertical):")
print(f"  initial error {err[0]*1e3:6.1f} mrad -> settled {err[-200:].max()*1e3:.3e} mrad")
print(f"  peak |tau|    {peak:6.3f} N.m")

try:
    import matplotlib.pyplot as plt
    import plotting
    plotting.setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.5))
    ax[0].plot(Xref[:, 0], Xref[:, 1], "k--", lw=0.8, label="reference")
    ax[0].plot(Xe[:, 0], Xe[:, 1], color="tab:blue", lw=0.9, label="executed")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel("$y$ [m]")
    ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title("general (asymmetric) robot", fontsize=8)
    ax[1].semilogy(t, np.maximum(err, 1e-9), color="tab:blue", lw=0.9)
    ax[1].set_xlabel("$t$ [s]"); ax[1].set_ylabel(r"$\|q-q_d\|$ [rad]")
    ax[1].set_title("computed-torque convergence", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "example_general_robot.pdf"))
    print("  plot saved: example_general_robot.pdf")
except ImportError:
    pass
