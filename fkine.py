# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
fkine.py -- FORWARD kinematics: given the two motor angles, where is the
end-effector?

For a five-bar this is the hard direction, because the answer is not unique.
Once the motors fix the two elbows, the end-effector must lie a distance l3
from elbow E1 AND a distance l4 from elbow E2 -- i.e. on the intersection of
two circles, which generically has TWO solutions.  Those two solutions are
the "assembly modes", labelled sigma = +1 and sigma = -1 (end-effector above
or below the line joining the elbows).

This file provides:
  * distal_angles, constraint : small forward helpers used elsewhere,
  * fk                        : the closed-form forward solution,
  * FKNewtonEstimator         : a per-sample forward-kinematics estimator that
                                turns measured joint angles into the
                                end-effector pose (and velocity), warm-started
                                from the previous sample.
"""

from dataclasses import dataclass
import numpy as np

from .robotgeometry import elbows
from .jacobians import AB, velocity_fk, singular_values


def distal_angles(geom, q, X):
    """Absolute orientations phi_i of the two distal links.  phi_i is the
    angle of the vector from elbow E_i to the end-effector X, measured CCW
    from +x.  Returns the array [phi1, phi2]."""
    b = np.asarray(X, float) - elbows(geom, q)
    return np.arctan2(b[:, 1], b[:, 0])


def constraint(geom, q, X):
    """Loop-closure residual in squared form.

    The mechanism is "assembled" exactly when the end-effector is the right
    distance from each elbow.  We measure the violation as
        c_i = 1/2 * ( |X - E_i|^2 - l_{2+i}^2 ),  i = 1, 2.
    Squaring avoids a square root and keeps everything polynomial, which is
    why the Newton solvers use this form.  Both c_i = 0 means the
    configuration is consistent.  Near a solution c_i is approximately
    l_{2+i} times the leg-length error, which is why solver tolerances divide
    by the distal length to obtain a meters-scale number.

    Returns the array [c1, c2].
    """
    b = np.asarray(X, float) - elbows(geom, q)
    return 0.5 * (np.einsum("ij,ij->i", b, b) - geom.distal ** 2)


def fk(geom, q, sigma=+1):
    """Closed-form forward kinematics.

    Steps:
      1. Place the elbows E1, E2 from the motor angles.
      2. Let P = E2 - E1 be the chord between them, D = |P|.
      3. The end-effector's foot on the chord is at distance
             e = (D^2 + l3^2 - l4^2) / (2 D)
         from E1 (for the symmetric robot this is the midpoint, e = D/2).
      4. Step off the chord by h = sqrt(l3^2 - e^2) along the unit normal, on
         the side chosen by sigma:
             X = E1 + e * P_hat + sigma * h * n_hat.

    Parameters
    ----------
    geom : FiveBarGeometry
    q : array-like (2,)   motor angles
    sigma : +1 or -1      assembly mode (which circle intersection)

    Returns
    -------
    (2,) array, the end-effector position X.

    Raises
    ------
    ValueError
        If the elbows coincide, or if the two distal circles do not meet
        (the existence condition |l3 - l4| <= D <= l3 + l4 fails), which
        happens at and beyond a Type-2 singularity.
    """
    E = elbows(geom, q)
    P = E[1] - E[0]
    D = float(np.hypot(P[0], P[1]))
    if D < 1e-12:
        raise ValueError("the two elbows coincide; forward kinematics is "
                         "undefined here")
    e = (D * D + geom.l3 ** 2 - geom.l4 ** 2) / (2.0 * D)
    h_squared = geom.l3 ** 2 - e * e
    if h_squared < -1e-12:
        raise ValueError("no assembly: the distal circles do not intersect "
                         "(need |l3 - l4| <= D <= l3 + l4); D = %.4f" % D)
    h = np.sqrt(max(h_squared, 0.0))
    P_hat = P / D
    n_hat = np.array([-P_hat[1], P_hat[0]])   # P rotated +90 degrees
    return E[0] + e * P_hat + sigma * h * n_hat


@dataclass
class FKResult:
    """What the forward-kinematics estimator returns for one sample."""
    X: np.ndarray              # estimated end-effector position
    Xdot: np.ndarray | None    # estimated velocity (None if no qdot given)
    iterations: int
    flag: bool                 # True if the iteration budget ran out
    residual: float            # meters (leg-length scale)
    det_A: float = 0.0         # live monitor: sign = assembly mode
    inv_cond: float = 0.0      # live monitor: 1/kappa(J) at the estimate


class FKNewtonEstimator:
    """Forward kinematics from measured joints, by warm-started Newton.

    Intended to be called once per control sample.  The assembly mode sigma0
    enters ONLY through the very first seed; afterwards each call warm-starts
    from the previous estimate, so the assembly mode is carried implicitly and
    never re-decided.  Each update also reports the live conditioning monitors
    (det A and 1/kappa at the estimate), handy for online safety checks.

    Parameters
    ----------
    geom : FiveBarGeometry
    sigma0 : +1 or -1   assembly mode used only for the initial seed.
    eps : stop tolerance (meters, leg-length scale).
    n_max : maximum Newton iterations per sample.
    mu : optional damping for ill-conditioned steps (0 = exact Newton).
    """

    def __init__(self, geom, sigma0=+1, eps=1e-10, n_max=10, mu=0.0):
        self.geom = geom
        self.sigma0 = sigma0
        self.eps = eps
        self.n_max = n_max
        self.mu = mu
        self._X = None        # last estimate (the warm start)

    def reset(self, q_meas=None):
        """Forget the warm start.  If q_meas is given, seed from the
        closed-form forward kinematics in assembly mode sigma0."""
        self._X = None if q_meas is None else fk(self.geom, q_meas,
                                                 self.sigma0)

    def update(self, q_meas, qdot_meas=None):
        """One estimation step from measured joint angles (and optional
        measured rates for the velocity estimate)."""
        q = np.asarray(q_meas, float)
        if self._X is None:
            self._X = fk(self.geom, q, self.sigma0)
        X = self._X.copy()
        E = elbows(self.geom, q)
        ld = self.geom.distal
        iters = 0
        residual = np.inf
        for iters in range(1, self.n_max + 1):
            b = X - E
            c = 0.5 * (np.einsum("ij,ij->i", b, b) - ld ** 2)
            residual = float(np.sum(np.abs(c) / ld))
            if residual <= self.eps:
                break
            A = b                       # the constraint Jacobian d c / d X
            if self.mu == 0.0:
                X = X - np.linalg.solve(A, c)
            else:
                X = X - A.T @ np.linalg.solve(
                    A @ A.T + self.mu ** 2 * np.eye(2), c)
        self._X = X
        Xdot = None
        if qdot_meas is not None:
            Xdot = velocity_fk(self.geom, q, X, qdot_meas)
        # live monitors at the estimate
        A, B, _ = AB(self.geom, q, X)
        dA = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
        ic = 0.0
        if abs(dA) > 1e-15:
            s_max, s_min = singular_values(np.linalg.solve(A, B))
            ic = s_min / s_max if s_max > 0 else 0.0
        budget_hit = iters >= self.n_max and residual > self.eps
        return FKResult(X.copy(), Xdot, iters, budget_hit, residual, dA, ic)
