# polypantograph example -- Author: Abolfazl Mohebbi, PhD., Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
paper_figures_dynamics.py -- regenerate the five dynamics figures of the
corresponding article end to end through the library: parameters from the URDF,
the model from dynamics.py, the controllers built on its inverse/forward dynamics.
Output goes to ./figures/.
"""

import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robotgeometry import elbows
from fkine import fk
from ikine import ik
from jacobians import jacobian
from urdf_loader import build_dynamics

USETEX = shutil.which("latex") is not None
plt.rcParams.update({"font.size": 8, "font.family": "serif", "text.usetex": USETEX,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "figure.dpi": 150})
if not USETEX:
    plt.rcParams["mathtext.fontset"] = "cm"

HERE = os.path.dirname(__file__)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
OMEGA, SIGMA, RATE = (+1, -1), +1, 1000.0
dt = 1.0 / RATE
URDF = os.path.join(HERE, "fivebar.urdf")
dyn_v, geom = build_dynamics(URDF, gravity=(0.0, -9.81), sigma=SIGMA)   # vertical
dyn_h, _ = build_dynamics(URDF, gravity=(0.0, 0.0), sigma=SIGMA)        # horizontal


def draw_robot(ax, q, X, color="0.6"):
    E = elbows(geom, q)
    for i in range(2):
        ax.plot([geom.bases[i, 0], E[i, 0], X[0]],
                [geom.bases[i, 1], E[i, 1], X[1]], "-", color=color, lw=1.0, zorder=1)
    ax.plot(geom.bases[:, 0], geom.bases[:, 1], "ks", ms=4, zorder=2)
    ax.plot(E[:, 0], E[:, 1], "o", color="0.45", ms=3, zorder=2)


def fig_path(name):
    return os.path.join(HERE, "figures", name)


# ================= DYN1: computed-torque tracking (vertical) =============
T = 3.0; N = int(T*RATE); t = np.arange(N)*dt; w = 2*np.pi/T
Xref = np.stack([0.06*np.cos(w*t), 0.30 + 0.06*np.sin(w*t)], axis=1)
Qref = np.array([ik(geom, X, OMEGA) for X in Xref])
Qd = np.gradient(Qref, dt, axis=0); Qdd = np.gradient(Qd, dt, axis=0)
Kp, Kd = 400.0, 40.0
q = Qref[0] + np.array([0.05, -0.04]); qd = np.zeros(2)
trk = {"err": [], "tau": [], "X": [], "q": []}
for k in range(N):
    e = Qref[k] - q
    aq = Qdd[k] + Kd*(Qd[k]-qd) + Kp*e
    tau = dyn_v.inverse_dynamics(q, qd, aq)
    q, qd = dyn_v.step(q, qd, tau, dt)
    trk["err"].append(np.linalg.norm(e)); trk["tau"].append(tau.copy())
    trk["X"].append(fk(geom, q, SIGMA)); trk["q"].append(q.copy())
for k in trk: trk[k] = np.array(trk[k])
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
ax[0].plot(Xref[:, 0], Xref[:, 1], "k--", lw=0.8, label="reference")
ax[0].plot(trk["X"][:, 0], trk["X"][:, 1], color="tab:blue", lw=0.9, label="executed")
draw_robot(ax[0], trk["q"][-1], trk["X"][-1])
ax[0].set_aspect("equal"); ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel("$y$ [m]")
ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
ax[0].set_title("computed-torque tracking", fontsize=8)
ax[1].semilogy(t, np.maximum(trk["err"], 1e-9), color="tab:blue", lw=0.9)
ax[1].set_xlabel("$t$ [s]"); ax[1].set_ylabel(r"$\|q-q_d\|$ [rad]")
ax[1].set_title("tracking error", fontsize=8)
ax[2].plot(t, trk["tau"][:, 0], color="tab:blue", lw=0.9, label=r"$\tau_1$")
ax[2].plot(t, trk["tau"][:, 1], color="tab:orange", lw=0.9, label=r"$\tau_2$")
ax[2].set_xlabel("$t$ [s]"); ax[2].set_ylabel(r"$\tau_i$ [N$\cdot$m]")
ax[2].legend(frameon=False, fontsize=6.5); ax[2].set_title("actuator torques", fontsize=8)
fig.tight_layout(); fig.savefig(fig_path("dyn1.pdf")); plt.close(fig)
print(f"DYN1 computed torque: {trk['err'][0]*1e3:.1f} -> {trk['err'][-200:].max()*1e3:.2e} mrad, "
      f"peak |tau| {np.abs(trk['tau']).max():.3f} N.m")

# ================= DYN2: forward-dynamics free coast (horizontal) ========
q0 = np.array([1.95, 1.19]); qd0 = np.array([1.2, 1.2])
st = np.concatenate([q0, qd0]); dt2 = 2e-4; N2 = int(0.8/dt2)
def deriv(s): return np.concatenate([s[2:], dyn_h.forward_dynamics(s[:2], s[2:], np.zeros(2))])
coast = {"X": [fk(geom, q0, SIGMA)], "q": [q0.copy()],
         "T": [dyn_h.kinetic_energy(q0, qd0)], "t": [0.0]}
for k in range(N2):
    k1 = deriv(st); k2 = deriv(st+dt2/2*k1); k3 = deriv(st+dt2/2*k2); k4 = deriv(st+dt2*k3)
    st = st + dt2/6*(k1+2*k2+2*k3+k4)
    coast["X"].append(fk(geom, st[:2], SIGMA)); coast["q"].append(st[:2].copy())
    coast["T"].append(dyn_h.kinetic_energy(st[:2], st[2:])); coast["t"].append((k+1)*dt2)
for k in coast: coast[k] = np.array(coast[k])
Edrift = (coast["T"].max()-coast["T"].min())/coast["T"][0]
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.3))
draw_robot(ax[0], coast["q"][0], coast["X"][0], color="0.82")
draw_robot(ax[0], coast["q"][-1], coast["X"][-1], color="0.45")
ax[0].plot(coast["X"][:, 0], coast["X"][:, 1], color="tab:blue", lw=1.3, zorder=3)
ax[0].plot(coast["X"][0, 0], coast["X"][0, 1], "o", color="tab:green", ms=6, zorder=4, label="start")
ax[0].plot(coast["X"][-1, 0], coast["X"][-1, 1], "s", color="tab:red", ms=6, zorder=4, label="end")
ax[0].set_aspect("equal"); ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel("$y$ [m]")
ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
ax[0].set_title("free coast (no torque, horizontal)", fontsize=8)
ax[1].plot(coast["t"], coast["T"]*1e3, color="tab:blue", lw=1.0)
ax[1].set_xlabel("$t$ [s]"); ax[1].set_ylabel("kinetic energy [mJ]")
ax[1].set_ylim(coast["T"][0]*1e3*0.98, coast["T"][0]*1e3*1.02)
ax[1].set_title("energy conservation (FD check)", fontsize=8)
fig.tight_layout(); fig.savefig(fig_path("dyn2.pdf")); plt.close(fig)
print(f"DYN2 free coast: relative energy drift over 0.8 s = {Edrift:.2e}")

# ================= DYN3: quasi-static loading, horizontal vs vertical =====
F = np.array([5.0, 0.0]); z = np.zeros(2)
tau_h = np.array([dyn_h.inverse_dynamics(ik(geom, X, OMEGA), z, z, wrench=F) for X in Xref])
tau_v = np.array([dyn_v.inverse_dynamics(ik(geom, X, OMEGA), z, z, wrench=F) for X in Xref])
tau_g = np.array([dyn_v.gravity_torque(ik(geom, X, OMEGA)) for X in Xref])
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.3))
ax[0].plot(t, tau_h[:, 0], color="tab:blue", lw=0.9, label=r"$\tau_1$")
ax[0].plot(t, tau_h[:, 1], color="tab:orange", lw=0.9, label=r"$\tau_2$")
ax[0].set_xlabel("$t$ [s]"); ax[0].set_ylabel(r"$\tau_i$ [N$\cdot$m]")
ax[0].legend(frameon=False, fontsize=6.5); ax[0].set_title("horizontal: tip force only", fontsize=8)
ax[1].plot(t, tau_v[:, 0], color="tab:blue", lw=0.9, label=r"$\tau_1$")
ax[1].plot(t, tau_v[:, 1], color="tab:orange", lw=0.9, label=r"$\tau_2$")
ax[1].plot(t, tau_g[:, 0], color="tab:blue", lw=0.7, ls=":", label="gravity only")
ax[1].plot(t, tau_g[:, 1], color="tab:orange", lw=0.7, ls=":")
ax[1].set_xlabel("$t$ [s]"); ax[1].set_ylabel(r"$\tau_i$ [N$\cdot$m]")
ax[1].legend(frameon=False, fontsize=6.5); ax[1].set_title("vertical: gravity + tip force", fontsize=8)
fig.tight_layout(); fig.savefig(fig_path("dyn3.pdf")); plt.close(fig)
print(f"DYN3 loading: peak |tau| horizontal {np.abs(tau_h).max():.3f}, "
      f"vertical {np.abs(tau_v).max():.3f}, gravity-only {np.abs(tau_g).max():.3f} N.m")

# ================= DYN4: hybrid force / position control (vertical) ======
xw, ydes, ke = 0.0, 0.30, 3000.0
Kpf, Kif, Bv, Kpy, Kdy = 0.8, 45.0, 95.0, 3000.0, 100.0
Nf = int(2.5*RATE); tf = np.arange(Nf)/RATE
q = ik(geom, np.array([xw, ydes]), OMEGA); qd = np.zeros(2); If = 0.0
fc = {"f": [], "fd": [], "x": [], "y": [], "tau": []}
for tk in tf:
    fdes = 0.0 if tk < 0.3 else (6.0 if tk < 1.5 else 10.0)
    X = fk(geom, q, SIGMA); J = jacobian(geom, q, X); Xdot = J @ qd
    fmeas = ke*max(X[0]-xw, 0.0)
    ef = fdes - fmeas; If += ef/RATE
    Fx = Kpf*ef + Kif*If - Bv*Xdot[0]
    Fy = Kpy*(ydes - X[1]) - Kdy*Xdot[1]
    tau = dyn_v.gravity_torque(q) + J.T @ np.array([Fx, Fy])
    qd = qd + dyn_v.forward_dynamics(q, qd, tau, wrench=np.array([-fmeas, 0.0]))/RATE
    q = q + qd/RATE
    fc["f"].append(fmeas); fc["fd"].append(fdes); fc["x"].append(X[0]); fc["y"].append(X[1]); fc["tau"].append(tau.copy())
for k in fc: fc[k] = np.array(fc[k])
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
ax[0].plot(tf, fc["fd"], "k--", lw=0.9, label="$f_d$"); ax[0].plot(tf, fc["f"], color="tab:blue", lw=1.0, label="$f$ measured")
ax[0].set_xlabel("$t$ [s]"); ax[0].set_ylabel("contact force [N]"); ax[0].legend(frameon=False, fontsize=6.5)
ax[0].set_title(r"force tracking ($x$)", fontsize=8)
ax[1].plot(tf, (fc["y"]-ydes)*1e3, color="tab:green", lw=1.0, label="$y-y_d$")
ax[1].plot(tf, (fc["x"]-xw)*1e3, color="tab:orange", lw=1.0, label="$x-x_w$")
ax[1].set_xlabel("$t$ [s]"); ax[1].set_ylabel("deviation [mm]"); ax[1].legend(frameon=False, fontsize=6.5)
ax[1].set_title(r"$y$ held, $x$ into wall", fontsize=8)
ax[2].plot(tf, fc["tau"][:, 0], color="tab:blue", lw=0.9, label=r"$\tau_1$")
ax[2].plot(tf, fc["tau"][:, 1], color="tab:orange", lw=0.9, label=r"$\tau_2$")
ax[2].set_xlabel("$t$ [s]"); ax[2].set_ylabel(r"$\tau_i$ [N$\cdot$m]"); ax[2].legend(frameon=False, fontsize=6.5)
ax[2].set_title("actuator torques", fontsize=8)
fig.tight_layout(); fig.savefig(fig_path("dyn4.pdf")); plt.close(fig)
print(f"DYN4 force control: steady force {fc['f'][-1]:.2f} N (target 10), max |y-y_d| {np.abs(fc['y']-ydes).max()*1e3:.2f} mm")

# ================= DYN5: impedance control on a line + disturbance =======
p0 = np.array([-0.06, 0.30]); p1 = np.array([0.06, 0.30]); Tline = 3.0
Kd_imp = np.array([800.0, 500.0]); Dd = np.array([12.0, 8.0])
Fdist = np.array([0.0, -12.0]); win = (1.5, 1.8)
Ni = int(4.0*RATE); ti = np.arange(Ni)/RATE
q = ik(geom, p0, OMEGA); qd = np.zeros(2)
imp = {"X": [], "Xd": [], "ey": [], "tau": []}
for tk in ti:
    s = min(max(tk/Tline, 0.0), 1.0)
    sp = 10*s**3 - 15*s**4 + 6*s**5; spd = (30*s**2 - 60*s**3 + 30*s**4)/Tline
    Xd = p0 + sp*(p1-p0); Xdd = spd*(p1-p0)
    X = fk(geom, q, SIGMA); J = jacobian(geom, q, X); Xdot = J @ qd
    eX = Xd - X; eXd = Xdd - Xdot
    tau = dyn_v.gravity_torque(q) + J.T @ (Kd_imp*eX + Dd*eXd)
    Fext = Fdist if (win[0] <= tk < win[1]) else np.zeros(2)
    qd = qd + dyn_v.forward_dynamics(q, qd, tau, wrench=Fext)/RATE
    q = q + qd/RATE
    imp["X"].append(X); imp["Xd"].append(Xd); imp["ey"].append(eX[1]); imp["tau"].append(tau.copy())
for k in imp: imp[k] = np.array(imp[k])
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
ax[0].plot(imp["Xd"][:, 0], imp["Xd"][:, 1], "k--", lw=0.9, label="reference line")
ax[0].plot(imp["X"][:, 0], imp["X"][:, 1], color="tab:blue", lw=1.0, label="executed")
ax[0].set_aspect("equal"); ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel("$y$ [m]")
ax[0].legend(frameon=False, fontsize=6.5); ax[0].set_title("straight-line tracking", fontsize=8)
ax[1].axvspan(win[0], win[1], color="tab:red", alpha=0.12, label="disturbance")
ax[1].plot(ti, imp["ey"]*1e3, color="tab:blue", lw=1.0)
ax[1].set_xlabel("$t$ [s]"); ax[1].set_ylabel(r"$y_d-y$ [mm]"); ax[1].legend(frameon=False, fontsize=6.5)
ax[1].set_title("deflection and spring-back", fontsize=8)
ax[2].plot(ti, imp["tau"][:, 0], color="tab:blue", lw=0.9, label=r"$\tau_1$")
ax[2].plot(ti, imp["tau"][:, 1], color="tab:orange", lw=0.9, label=r"$\tau_2$")
ax[2].set_xlabel("$t$ [s]"); ax[2].set_ylabel(r"$\tau_i$ [N$\cdot$m]"); ax[2].legend(frameon=False, fontsize=6.5)
ax[2].set_title("actuator torques", fontsize=8)
fig.tight_layout(); fig.savefig(fig_path("dyn5.pdf")); plt.close(fig)
rel = int(win[1]*RATE); peakdef = np.abs(imp["ey"][int(win[0]*RATE):rel]).max()
print(f"DYN5 impedance: peak deflection {peakdef*1e3:.1f} mm")
print("figures written to ./figures/")
