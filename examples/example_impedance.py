# polypantograph example -- Author: Abolfazl Mohebbi, PhD., Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
example_impedance.py -- impedance control of the five-bar on three task-space
references, each probed by an end-effector perturbation (a "human push").

The controller renders a spring-damper about the reference,
    tau = g(q) + J^T ( Kd (Xd - X) + Dd (Xd_dot - X_dot) ),
built on the dynamic model loaded from the URDF.  Three references are run
separately:
  1. a straight line,
  2. a curved (circular) path,
  3. a hold (zero displacement -- the reference never moves).
For each, a step force is applied at the end-effector over a fixed window and
then released; the apex deflects and springs back according to the rendered
impedance.

Outputs (saved next to this script): one PDF per trajectory plus a combined
overview, each showing the xy path, x(t) and y(t), q1(t) and q2(t), and the
tracking error with the perturbation window shaded.

To move this to a real robot: read q from your RobotIO subclass, send tau to
current-controlled drives, and obtain (Xd, Xd_dot) from your task planner; the
"perturbation" is then simply whatever the human applies.
"""

import numpy as np
from ikine import ik
from fkine import fk
from jacobians import jacobian
from urdf_loader import build_dynamics

HERE = os.path.dirname(__file__)
OMEGA, SIGMA, RATE = (+1, -1), +1, 1000.0
dt = 1.0 / RATE

# --- mounting (choose here) ---------------------------------------------
GRAVITY = (0.0, -9.81)          # vertical, gravity compensated by the law
# GRAVITY = (0.0, 0.0)          # horizontal alternative (no gravity)

# --- symmetric uniform robot ---
# dyn, geom = build_dynamics(os.path.join(HERE, "fivebar.urdf"), gravity=GRAVITY, sigma=SIGMA)

# --- general robot ---
dyn, geom = build_dynamics(os.path.join(HERE, "general_arm.urdf"), gravity=(0.0, -9.81))

# --- impedance gains and perturbation -----------------------------------
Kd = np.array([800.0, 500.0])   # rendered stiffness [N/m] (x, y)
Dd = np.array([12.0, 8.0])      # rendered damping  [N.s/m] (x, y), underdamped
F_PERT = np.array([6.0, -10.0]) # step "human push" at the apex [N]
WIN = (1.5, 1.8)                # held over this window, then released
T, TMOVE = 4.0, 3.0             # total time and motion time (then hold)
N = int(T * RATE); t = np.arange(N) * dt


def _minjerk(s):
    return 10*s**3 - 15*s**4 + 6*s**5, 30*s**2 - 60*s**3 + 30*s**4


def ref_line(tk):
    p0, p1 = np.array([-0.06, 0.30]), np.array([0.06, 0.30])
    s = min(max(tk/TMOVE, 0.0), 1.0); sp, spd = _minjerk(s)
    return p0 + sp*(p1-p0), (spd/TMOVE)*(p1-p0)


def ref_curve(tk):
    c, r = np.array([0.0, 0.30]), 0.05
    s = min(max(tk/TMOVE, 0.0), 1.0); sp, spd = _minjerk(s)
    th = 2*np.pi*sp; thd = 2*np.pi*spd/TMOVE
    return (c + r*np.array([np.cos(th), np.sin(th)]),
            r*thd*np.array([-np.sin(th), np.cos(th)]))


def ref_hold(tk):
    return np.array([0.0, 0.30]), np.zeros(2)


def run(ref):
    """Run impedance control on one reference; return a log of arrays."""
    q = ik(geom, ref(0.0)[0], OMEGA); qd = np.zeros(2)
    rec = {k: [] for k in ("x", "y", "xd", "yd", "q1", "q2",
                           "e", "fx", "fy", "tau1", "tau2")}
    for k in range(N):
        Xd, Xdd = ref(t[k])
        X = fk(geom, q, SIGMA); J = jacobian(geom, q, X); Xdot = J @ qd
        eX = Xd - X; eXd = Xdd - Xdot
        tau = dyn.gravity_torque(q) + J.T @ (Kd*eX + Dd*eXd)
        Fp = F_PERT if (WIN[0] <= t[k] < WIN[1]) else np.zeros(2)
        qd = qd + dyn.forward_dynamics(q, qd, tau, wrench=Fp) / RATE
        q = q + qd / RATE
        rec["x"].append(X[0]); rec["y"].append(X[1])
        rec["xd"].append(Xd[0]); rec["yd"].append(Xd[1])
        rec["q1"].append(q[0]); rec["q2"].append(q[1])
        rec["e"].append(np.linalg.norm(eX))
        rec["fx"].append(Fp[0]); rec["fy"].append(Fp[1])
        rec["tau1"].append(tau[0]); rec["tau2"].append(tau[1])
    return {k: np.array(v) for k, v in rec.items()}


CASES = [("straight line", ref_line), ("curved path", ref_curve),
         ("hold (zero)", ref_hold)]
LOGS = {name: run(ref) for name, ref in CASES}
for name, L in LOGS.items():
    peak = np.abs(L["e"][int(WIN[0]*RATE):int(WIN[1]*RATE)]).max()*1e3
    print(f"{name:14s}: peak deflection during push {peak:5.1f} mm, "
          f"final error {L['e'][-1]*1e3:.3f} mm")

# --- plotting -----------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import plotting
    plotting.setup_style()

    def panels(axrow, L, title):
        ax = axrow
        ax[0].plot(L["xd"], L["yd"], "k--", lw=0.8, label="reference")
        ax[0].plot(L["x"], L["y"], color="tab:blue", lw=1.0, label="executed")
        kp = int(WIN[0]*RATE)
        ax[0].plot(L["x"][kp], L["y"][kp], "o", color="tab:red", ms=4,
                   label="push start")
        ax[0].set_aspect("equal"); ax[0].set_xlabel("$x$ [m]")
        ax[0].set_ylabel("$y$ [m]"); ax[0].legend(frameon=False, fontsize=6)
        ax[0].set_title(f"{title}: path", fontsize=8)

        ax[1].axvspan(*WIN, color="tab:red", alpha=0.12)
        ax[1].plot(t, L["xd"], color="tab:blue", lw=0.7, ls="--")
        ax[1].plot(t, L["x"], color="tab:blue", lw=1.0, label="$x$")
        ax[1].plot(t, L["yd"], color="tab:orange", lw=0.7, ls="--")
        ax[1].plot(t, L["y"], color="tab:orange", lw=1.0, label="$y$")
        ax[1].set_xlabel("$t$ [s]"); ax[1].set_ylabel("position [m]")
        ax[1].legend(frameon=False, fontsize=6, ncol=2)
        ax[1].set_title("end-effector $x,y$ vs $t$", fontsize=8)

        ax[2].axvspan(*WIN, color="tab:red", alpha=0.12)
        ax[2].plot(t, np.degrees(L["q1"]), color="tab:green", lw=1.0,
                   label="$q_1$")
        ax[2].plot(t, np.degrees(L["q2"]), color="tab:purple", lw=1.0,
                   label="$q_2$")
        ax[2].set_xlabel("$t$ [s]"); ax[2].set_ylabel("joint angle [deg]")
        ax[2].legend(frameon=False, fontsize=6)
        ax[2].set_title("joints $q_1,q_2$ vs $t$", fontsize=8)

        ax[3].axvspan(*WIN, color="tab:red", alpha=0.12, label="push")
        ax[3].plot(t, L["e"]*1e3, color="tab:blue", lw=1.0)
        ax[3].set_xlabel("$t$ [s]"); ax[3].set_ylabel("track error [mm]")
        ax[3].legend(frameon=False, fontsize=6)
        ax[3].set_title("tracking error", fontsize=8)

    # per-trajectory PDFs
    short = {"straight line": "line", "curved path": "curve",
             "hold (zero)": "hold"}
    for name, L in LOGS.items():
        fig, ax = plt.subplots(1, 4, figsize=(11, 2.5))
        panels(ax, L, name)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"example_impedance_{short[name]}.pdf"))
        plt.close(fig)

    # combined overview
    fig, axes = plt.subplots(3, 4, figsize=(11, 7.2))
    for r, (name, L) in enumerate(LOGS.items()):
        panels(axes[r], L, name)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "example_impedance_overview.pdf"))
    plt.close(fig)
    print("saved: example_impedance_{line,curve,hold,overview}.pdf")
except ImportError:
    print("(matplotlib not installed -- skipped plots)")
