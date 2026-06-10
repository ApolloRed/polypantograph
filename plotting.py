# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
plotting.py -- optional figures from a run's LoopLog (needs matplotlib).

Each function takes the records produced by TrackingController and draws the
corresponding figure.  matplotlib is imported only when these functions are
called, so the rest of the library has no plotting dependency.  Fonts use
real LaTeX (Computer Modern) when a 'latex' program is on the PATH, and
matplotlib's built-in Computer-Modern math font otherwise.
"""

import shutil
import numpy as np

from robotgeometry import elbows
from ikine import ik
from fkine import fk
from jacobians import jacobian, singular_values

# colors used to tell the three solvers apart in comparison plots
_COLORS = {"(a) closed form": "tab:green",
           "(b) task Newton": "tab:orange",
           "(c) per-leg Newton": "tab:blue"}


def _plt():
    """Import matplotlib lazily, with a clear message if it is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError("plotting needs matplotlib (pip install matplotlib)")             from e


def setup_style(usetex="auto"):
    """ usetex='auto' uses LaTeX if available.
    Returns True if LaTeX text rendering is on."""
    plt = _plt()
    if usetex == "auto":
        usetex = shutil.which("latex") is not None
    plt.rcParams.update({"font.size": 8, "font.family": "serif",
                         "text.usetex": bool(usetex),
                         "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                         "figure.dpi": 150})
    if not usetex:
        plt.rcParams["mathtext.fontset"] = "cm"
    return bool(usetex)


def draw_robot(ax, geom, q, X, color="0.6"):
    """Sketch one linkage posture (both legs, motors, elbows) on an axis."""
    E = elbows(geom, q)
    X = np.asarray(X, float)
    for i in range(2):
        ax.plot([geom.bases[i, 0], E[i, 0], X[0]],
                [geom.bases[i, 1], E[i, 1], X[1]],
                "-", color=color, lw=1.0, zorder=1)
    ax.plot(geom.bases[:, 0], geom.bases[:, 1], "ks", ms=4, zorder=2)
    ax.plot(E[:, 0], E[:, 1], "o", color="0.45", ms=3, zorder=2)


def _solver_error(geom, log, sigma):
    """Per-sample |FK(q_cmd) - X_cmd|: the solver's own tracking error."""
    e = np.empty(len(log.t))
    for k in range(len(log.t)):
        e[k] = np.linalg.norm(fk(geom, log.q_cmd[k], sigma) - log.X_cmd[k])
    return e


def plot_tracking_comparison(geom, logs, sigma=+1):
    """Section 9.3 figure: end-effector path | joint angles | tracking error,
    for a dict {name: LoopLog} of solver runs on the same path."""
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ref = next(iter(logs.values()))
    ax[0].plot(ref.X_des[:, 0], ref.X_des[:, 1], "k--", lw=0.8,
               label="reference")
    ax[0].plot(ref.X_cmd[:, 0], ref.X_cmd[:, 1], color="tab:blue", lw=0.9,
               label="executed (all)")
    draw_robot(ax[0], geom, ref.q_cmd[0], ref.X_cmd[0])
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel("$y$ [m]")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    ax[0].set_title("end-effector path", fontsize=8)
    for name, log in logs.items():
        c = _COLORS.get(name)
        ax[1].plot(log.t, np.degrees(log.q_cmd[:, 0]), color=c, lw=0.9)
        ax[1].plot(log.t, np.degrees(log.q_cmd[:, 1]), color=c, lw=0.9, ls=":")
    ax[1].set_xlabel("$t$ [s]"); ax[1].set_ylabel("$q_i$ [deg]")
    ax[1].set_title("joint trajectories (overlaid)", fontsize=8)
    for name, log in logs.items():
        e = _solver_error(geom, log, sigma)
        ax[2].semilogy(log.t, np.maximum(e, 1e-17), color=_COLORS.get(name),
                       lw=0.9, label=name)
    ax[2].set_xlabel("$t$ [s]")
    ax[2].set_ylabel(r"$\|X(q[k])-X_d[k]\|$ [m]")
    ax[2].set_title("tracking error", fontsize=8)
    ax[2].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    return fig


def plot_wall_comparison(geom, logs, wall, sigma=+1):
    """Section 9.4 figure: same layout as the tracking comparison, with the
    virtual-wall box drawn and the error taken against the projected
    reference."""
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ref = next(iter(logs.values()))
    rect = plt.Rectangle((wall.xmin, wall.ymin), wall.xmax - wall.xmin,
                         wall.ymax - wall.ymin, fill=False, ec="tab:red",
                         lw=1.0)
    ax[0].add_patch(rect)
    ax[0].plot(ref.X_des[:, 0], ref.X_des[:, 1], "k--", lw=0.8,
               label="reference")
    ax[0].plot(ref.X_cmd[:, 0], ref.X_cmd[:, 1], color="tab:blue", lw=1.0,
               label="executed (all)")
    draw_robot(ax[0], geom, ref.q_cmd[0], ref.X_cmd[0])
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel("$y$ [m]")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    ax[0].set_title(r"end-effector path, wall at $x\leq %g$ m" % wall.xmax,
                    fontsize=8)
    for name, log in logs.items():
        c = _COLORS.get(name)
        ax[1].plot(log.t, np.degrees(log.q_cmd[:, 0]), color=c, lw=0.9)
        ax[1].plot(log.t, np.degrees(log.q_cmd[:, 1]), color=c, lw=0.9, ls=":")
    ax[1].set_xlabel("$t$ [s]"); ax[1].set_ylabel("$q_i$ [deg]")
    ax[1].set_title("joint trajectories (overlaid)", fontsize=8)
    for name, log in logs.items():
        e = _solver_error(geom, log, sigma)
        ax[2].semilogy(log.t, np.maximum(e, 1e-17), color=_COLORS.get(name),
                       lw=0.9, label=name)
    ax[2].set_xlabel("$t$ [s]")
    ax[2].set_ylabel(r"$\|X(q[k])-\Pi_{\mathcal{C}}(X_d[k])\|$ [m]")
    ax[2].set_title("error vs projected reference", fontsize=8)
    ax[2].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    return fig


def _reach_boundary(ax, geom, y_min=-0.02, label=None):
    """Draw the true outer reach boundary (each leg's stretched circle masked
    to the other leg's disk -- a lens, symmetric for the symmetric robot)."""
    th = np.linspace(0, 2 * np.pi, 2000)
    first = True
    for i, j in ((0, 1), (1, 0)):
        ri = geom.reach_annulus(i)[1]
        rj = geom.reach_annulus(j)[1]
        P = geom.bases[i][None, :] + ri * np.stack(
            [np.cos(th), np.sin(th)], axis=1)
        keep = (np.linalg.norm(P - geom.bases[j][None, :], axis=1) <= rj)             & (P[:, 1] >= y_min)
        P = P.copy(); P[~keep] = np.nan
        ax.plot(P[:, 0], P[:, 1], color="0.7", lw=0.8,
                label=label if first else None)
        first = False


def plot_singularity_encounters(geom, log_type1, log_type2, kappa0,
                                reach_margin=0.005, omega=(+1, -1)):
    """Section 9.5 figure (2x2): Type-1 path and reach margin (top), Type-2
    path and singularity monitors (bottom).  The dashed 'commanded' monitor
    curves are recomputed from the raw reference stored in the logs."""
    plt = _plt()
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 4.6))
    _reach_boundary(ax[0, 0], geom, label="reach boundary")
    ax[0, 0].plot(log_type1.X_des[:, 0], log_type1.X_des[:, 1], "k--",
                  lw=0.8, label="reference")
    ax[0, 0].plot(log_type1.X_cmd[:, 0], log_type1.X_cmd[:, 1],
                  color="tab:blue", lw=1.0, label="executed")
    ax[0, 0].plot(geom.bases[:, 0], geom.bases[:, 1], "ks", ms=4)
    ax[0, 0].set_aspect("equal")
    ax[0, 0].legend(frameon=False, fontsize=6, loc="upper left")
    ax[0, 0].set_title("Type-1 encounter: path", fontsize=8)
    ax[0, 0].set_xlabel("$x$ [m]"); ax[0, 0].set_ylabel("$y$ [m]")
    rmax = geom.reach_annulus(0)[1]
    r1d = np.linalg.norm(log_type1.X_des - geom.bases[0], axis=1)
    r1e = np.linalg.norm(log_type1.X_cmd - geom.bases[0], axis=1)
    ax[0, 1].plot(log_type1.t, r1d, "k--", lw=0.8, label="$r_1$ commanded")
    ax[0, 1].plot(log_type1.t, r1e, color="tab:blue", lw=1.0,
                  label="$r_1$ executed")
    ax[0, 1].axhline(rmax, color="0.4", lw=0.8)
    ax[0, 1].axhline(rmax - reach_margin, color="tab:red", lw=0.8, ls=":")
    ax[0, 1].set_xlabel("$t$ [s]"); ax[0, 1].set_ylabel("$r_1$ [m]")
    ax[0, 1].legend(frameon=False, fontsize=6)
    ax[0, 1].set_title("reach margin", fontsize=8)
    ax[1, 0].plot(log_type2.X_des[:, 0], log_type2.X_des[:, 1], "k--",
                  lw=0.8, label="reference")
    ax[1, 0].plot(log_type2.X_cmd[:, 0], log_type2.X_cmd[:, 1],
                  color="tab:blue", lw=1.0, label="executed")
    ax[1, 0].plot(geom.bases[:, 0], geom.bases[:, 1], "ks", ms=4)
    ax[1, 0].set_aspect("equal")
    ax[1, 0].legend(frameon=False, fontsize=6, loc="upper right")
    ax[1, 0].set_xlabel("$x$ [m]"); ax[1, 0].set_ylabel("$y$ [m]")
    ax[1, 0].set_title("Type-2 encounter: path", fontsize=8)
    dA_des = np.empty(len(log_type2.t))
    for k in range(len(log_type2.t)):
        X = log_type2.X_des[k]
        try:
            q = ik(geom, X, omega)
            b = X - elbows(geom, q)
            dA_des[k] = b[0, 0] * b[1, 1] - b[0, 1] * b[1, 0]
        except ValueError:
            dA_des[k] = np.nan
    ax[1, 1].plot(log_type2.t, dA_des * 1e3, "k--", lw=0.8,
                  label=r"$\det A$ along reference")
    ax[1, 1].plot(log_type2.t, log_type2.det_A * 1e3, color="tab:blue",
                  lw=1.0, label=r"$\det A$ executed")
    ax[1, 1].axhline(0, color="0.4", lw=0.8)
    ax2 = ax[1, 1].twinx()
    ax2.plot(log_type2.t, log_type2.inv_cond, color="tab:green", lw=0.9)
    ax2.axhline(kappa0, color="tab:green", lw=0.8, ls=":")
    ax2.set_ylabel(r"$1/\kappa(J)$", color="tab:green")
    ax2.set_ylim(0, 0.8)
    ax[1, 1].set_xlabel("$t$ [s]")
    ax[1, 1].set_ylabel(r"$\det A$ [$10^{-3}$ m$^2$]")
    ax[1, 1].legend(frameon=False, fontsize=6, loc="lower center")
    ax[1, 1].set_title("singularity monitors", fontsize=8)
    fig.tight_layout()
    return fig


def plot_fk_quantized(geom, log, enc_resolution, sigma=+1):
    """Section 9.6 figure: quantized joint angles | reconstructed path |
    reconstruction error against the encoder-resolution bound."""
    plt = _plt()
    n = len(log.t)
    X_true = np.empty((n, 2))
    bound = np.empty(n)
    for k in range(n):
        X_true[k] = fk(geom, log.q_cmd[k], sigma)
        bound[k] = singular_values(
            jacobian(geom, log.q_cmd[k], X_true[k]))[0] * enc_resolution
    err = np.linalg.norm(log.X_exec - X_true, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ax[0].plot(log.t, np.degrees(log.q_meas[:, 0]), color="tab:blue", lw=0.9,
               label="$q_1$")
    ax[0].plot(log.t, np.degrees(log.q_meas[:, 1]), color="tab:blue", lw=0.9,
               ls=":", label="$q_2$")
    ax[0].set_xlabel("$t$ [s]"); ax[0].set_ylabel("$q_i$ [deg]")
    ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title("quantized joint measurements", fontsize=8)
    ax[1].plot(X_true[:, 0], X_true[:, 1], "k--", lw=0.8, label="true")
    ax[1].plot(log.X_exec[:, 0], log.X_exec[:, 1], color="tab:blue", lw=0.9,
               label=r"$\hat{X}$ from FK")
    draw_robot(ax[1], geom, log.q_meas[0], log.X_exec[0])
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("$x$ [m]"); ax[1].set_ylabel("$y$ [m]")
    ax[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    ax[1].set_title("reconstructed end-effector path", fontsize=8)
    ax[2].plot(log.t, err * 1e6, color="tab:blue", lw=0.8,
               label=r"$\|\hat{X}-X\|$")
    ax[2].plot(log.t, bound * 1e6, color="tab:red", lw=0.9, ls="--",
               label=r"$\|J\|_2\,\Delta q_{\rm enc}$")
    ax[2].set_xlabel("$t$ [s]"); ax[2].set_ylabel(r"error [$\mu$m]")
    ax[2].legend(frameon=False, fontsize=6.5)
    ax[2].set_title("reconstruction error", fontsize=8)
    fig.tight_layout()
    return fig
