# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
ikine.py -- INVERSE kinematics: given a desired end-effector position, what
motor angles put it there?

For a five-bar this is the easy direction, because the two legs decouple: each
is an independent two-link (2R) arm with a textbook closed-form solution.
Each leg has two ways to reach the target -- "elbow up" or "elbow down" --
encoded by a working-mode sign omega_i = +/-1, so the full mechanism has up to
2 x 2 = 4 inverse solutions.

This file provides:
  * ik, ik_branches, working_modes : the closed-form inverse and helpers,
  * three per-sample IK solvers for a running controller:
        ClosedFormSolver   - the algebraic solution with a continuity guard,
        TaskNewtonSolver   - Newton iteration on the end-effector error,
        PerLegNewtonSolver - the recommended in-loop solver: forward-
                             kinematics-free, per-leg damped Newton.
    All three share the call result = solver.solve(X_desired, q_warm_start),
    so a controller can swap one for another freely.
"""

from dataclasses import dataclass
import numpy as np

from robotgeometry import wrap, elbows
from fkine import fk, constraint
from jacobians import AB


# --------------------------------------------------------------------------
# Closed-form inverse kinematics and branch helpers
# --------------------------------------------------------------------------
def ik(geom, X, omega=(+1, -1), clip=False):
    """Closed-form inverse kinematics.

    For each leg i with motor at O_i and links (l_i proximal, l_{2+i} distal):
      * alpha_i = angle of the vector O_i -> X (the reach direction),
      * beta_i  = interior elbow angle from the law of cosines,
      * q_i     = alpha_i + omega_i * beta_i.

    Parameters
    ----------
    geom : FiveBarGeometry
    X : array-like (2,)        desired end-effector position
    omega : pair of +/-1       working mode of each leg (elbow side).
        The default (+1, -1) is the "elbows-out" posture.
    clip : bool
        If False (default), a target outside a leg's reachable annulus raises
        ValueError.  If True, the cosine is clamped to [-1, 1], returning the
        nearest boundary (fully stretched/folded) solution -- useful inside a
        solver that wants a graceful value rather than an exception.

    Returns
    -------
    (2,) array of motor angles, wrapped to (-pi, pi].
    """
    X = np.asarray(X, float)
    q = np.empty(2)
    for i in range(2):
        to_target = X - geom.bases[i]
        r = float(np.hypot(to_target[0], to_target[1]))
        lp = geom.proximal[i]
        ld = geom.distal[i]
        cos_beta = (r * r + lp * lp - ld * ld) / (2.0 * lp * r)
        if not clip and not (-1.0 - 1e-12 <= cos_beta <= 1.0 + 1e-12):
            raise ValueError(
                "target is outside the reachable annulus of leg %d "
                "(r = %.4f, annulus = %s)"
                % (i + 1, r, geom.reach_annulus(i)))
        beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
        alpha = np.arctan2(to_target[1], to_target[0])
        q[i] = alpha + omega[i] * beta
    return wrap(q)


def ik_branches(geom, X):
    """All inverse solutions that exist at X, as a dict keyed by the working
    mode pair.  Up to four entries; unreachable combinations are omitted."""
    solutions = {}
    for w1 in (+1, -1):
        for w2 in (+1, -1):
            try:
                solutions[(w1, w2)] = ik(geom, X, (w1, w2))
            except ValueError:
                pass
    return solutions


def working_modes(geom, q, X):
    """Recover the working-mode signs (omega1, omega2) of an existing
    configuration (q, X).  Because q_i = alpha_i + omega_i * beta_i with
    beta_i in (0, pi), the sign of the wrapped difference q_i - alpha_i is
    exactly omega_i.  Used at start-up to label the powered-up posture.
    Returns a tuple of two +/-1 values."""
    X = np.asarray(X, float)
    to_target = X - geom.bases
    alpha = np.arctan2(to_target[:, 1], to_target[:, 0])
    signs = np.sign(wrap(np.asarray(q, float) - alpha))
    return (int(signs[0]) if signs[0] != 0 else +1,
            int(signs[1]) if signs[1] != 0 else +1)


# --------------------------------------------------------------------------
# Per-sample inverse-kinematics solvers
# --------------------------------------------------------------------------
@dataclass
class SolveResult:
    """What an IK solver returns for one sample."""
    q: np.ndarray          # solved motor angles
    iterations: int        # loop passes used
    flag: bool             # True if the guard tripped or the budget ran out
    residual: float        # remaining error, meters (leg-length scale)


class ClosedFormSolver:
    """Inverse kinematics by the closed-form formula, with a safety guard.

    Needs the working modes (omega).  Any solution that jumps more than
    dq_guard radians from the previous sample is rejected (the previous
    command is held and the flag is raised), protecting against a
    working-mode flip if the algebra ever picks the wrong branch.
    """

    def __init__(self, geom, omega=(+1, -1), dq_guard=0.05):
        self.geom = geom
        self.omega = tuple(omega)
        self.dq_guard = dq_guard

    def solve(self, X_desired, q_warm_start):
        q = ik(self.geom, X_desired, self.omega, clip=True)
        if np.max(np.abs(wrap(q - q_warm_start))) > self.dq_guard:
            q_prev = np.asarray(q_warm_start, float).copy()
            return SolveResult(q_prev, 1, True, self._residual(q_prev,
                                                               X_desired))
        return SolveResult(q, 1, False, self._residual(q, X_desired))

    def _residual(self, q, X_desired):
        return float(np.sum(np.abs(constraint(self.geom, q, X_desired))
                            / self.geom.distal))


class TaskNewtonSolver:
    """Inverse kinematics by Newton iteration in task space.

    Iterates on the end-effector error e = X_desired - fk(q); the step solves
    B dq = A e (optionally damped).  Because it evaluates forward kinematics,
    it must be told the assembly mode sigma.
    """

    def __init__(self, geom, sigma=+1, eps=1e-9, n_max=10, mu=0.0):
        self.geom = geom
        self.sigma = sigma
        self.eps = eps
        self.n_max = n_max
        self.mu = mu

    def solve(self, X_desired, q_warm_start):
        X_desired = np.asarray(X_desired, float)
        q = np.asarray(q_warm_start, float).copy()
        iters = 0
        err = np.inf
        for iters in range(1, self.n_max + 1):
            try:
                X = fk(self.geom, q, self.sigma)
            except ValueError:
                # forward kinematics undefined (at/just past Type-2): bail out
                return SolveResult(np.asarray(q_warm_start, float).copy(),
                                   iters, True, np.inf)
            e = X_desired - X
            err = float(np.hypot(e[0], e[1]))
            if err <= self.eps:
                return SolveResult(wrap(q), iters, False, err)
            A, B, _ = AB(self.geom, q, X)
            if self.mu == 0.0:
                dq = np.linalg.solve(B, A @ e)
            else:
                J = np.linalg.solve(A, B)
                dq = J.T @ np.linalg.solve(J @ J.T + self.mu ** 2 * np.eye(2),
                                           e)
            q = q + dq
        return SolveResult(wrap(q), iters, True, err)


class PerLegNewtonSolver:
    """Inverse kinematics by a forward-kinematics-free per-leg damped Newton.

    The recommended solver for a real-time loop.  For each leg it drives the
    loop-closure residual c_i to zero with the update
        q_i <- q_i + (l_i k_i) c_i / ((l_i k_i)^2 + mu^2),
    where mu is an optional damping that keeps the step bounded as the leg
    approaches a fully stretched/folded (Type-1) posture (k_i -> 0).  It needs
    no assembly mode and no forward-kinematics call; the warm start carries
    all branch information.
    """

    def __init__(self, geom, eps=1e-9, n_max=10, mu=0.0):
        self.geom = geom
        self.eps = eps
        self.n_max = n_max
        self.mu = mu

    def solve(self, X_desired, q_warm_start):
        X_desired = np.asarray(X_desired, float)
        q = np.asarray(q_warm_start, float).copy()
        ld = self.geom.distal
        iters = 0
        residual = np.inf
        for iters in range(1, self.n_max + 1):
            b = X_desired - elbows(self.geom, q)
            c = 0.5 * (np.einsum("ij,ij->i", b, b) - ld ** 2)
            residual = float(np.sum(np.abs(c) / ld))
            if residual <= self.eps:
                return SolveResult(wrap(q), iters, False, residual)
            k = np.cos(q) * b[:, 1] - np.sin(q) * b[:, 0]
            ak = self.geom.proximal * k
            if self.mu == 0.0:
                dq = c / ak
            else:
                dq = ak * c / (ak * ak + self.mu ** 2)
            q = q + dq
        return SolveResult(wrap(q), iters, True, residual)
