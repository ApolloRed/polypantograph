# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
jacobians.py -- velocity kinematics and the quantities derived from it
(conditioning, singularities, the damped inverse).

Because the five-bar is a closed loop, its velocity relation is implicit:
differentiating the loop-closure constraint gives two matrices A and B with
        A(q, X) * Xdot = B(q, X) * qdot,
where
    A = [ b1^T ; b2^T ]              rows are the distal-link vectors b_i = X - E_i,
    B = diag(l1 k1, l2 k2),         k_i = (a_hat_i  x  b_i) = l_{2+i} sin(phi_i - q_i).
When A is invertible the end-effector Jacobian is J = A^{-1} B (Xdot = J qdot).

Two singularity families fall straight out of the determinants:
    det B = 0  -> Type-1 (a leg fully stretched/folded; workspace boundary),
    det A = 0  -> Type-2 (distal links collinear; interior, end-effector
                  gains an uncontrollable freedom).
Also det A = l3 l4 sin(phi2 - phi1) = sigma h D, so sign(det A) is the
assembly mode sigma.
"""

import numpy as np
from robotgeometry import elbows


def AB(geom, q, X):
    """Return the constraint matrices (A, B) and the helper vector k at the
    configuration (q, X).  See the module docstring for definitions."""
    q = np.asarray(q, float)
    b = np.asarray(X, float) - elbows(geom, q)        # rows b1, b2
    A = b.copy()
    # k_i = a_hat_i x b_i  (2-D scalar cross product)
    k = np.cos(q) * b[:, 1] - np.sin(q) * b[:, 0]
    B = np.diag(geom.proximal * k)
    return A, B, k


def jacobian(geom, q, X):
    """End-effector Jacobian J = A^{-1} B, so that Xdot = J qdot."""
    A, B, _ = AB(geom, q, X)
    return np.linalg.solve(A, B)


def jacobian_inverse(geom, q, X):
    """Inverse Jacobian J^{-1} = B^{-1} A.

    Because B is diagonal this needs no matrix inversion -- each row is just
    b_i^T / (l_i k_i).  It stays finite everywhere except Type-1
    (where some k_i -> 0); in particular it is well-behaved at Type-2, which
    is why velocity_ik below uses this form rather than inverting J.
    """
    A, _, k = AB(geom, q, X)
    return A / (geom.proximal * k)[:, None]


def det_A(geom, q, X):
    """det A = b1 x b2.  Its sign is the assembly mode sigma; its magnitude
    (= h D) measures how far the posture is from a Type-2 singularity."""
    b = np.asarray(X, float) - elbows(geom, q)
    return float(b[0, 0] * b[1, 1] - b[0, 1] * b[1, 0])


def det_B(geom, q, X):
    """det B = l1 l2 k1 k2.  Vanishes at a Type-1 singularity."""
    _, _, k = AB(geom, q, X)
    return float(geom.l1 * geom.l2 * k[0] * k[1])


def assembly_mode(geom, q, X):
    """The assembly-mode sign sigma = sign(det A) (+1 or -1)."""
    return +1 if det_A(geom, q, X) >= 0.0 else -1


def singular_values(M):
    """Singular values (s_max, s_min) of a 2x2 matrix, in closed form
    (faster and dependency-free compared to a general SVD call)."""
    G = M.T @ M
    trace = G[0, 0] + G[1, 1]
    det = abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    disc = np.sqrt(max(trace * trace - 4.0 * det * det, 0.0))
    s_max = np.sqrt(max((trace + disc) / 2.0, 0.0))
    s_min = np.sqrt(max((trace - disc) / 2.0, 0.0))
    return s_max, s_min


def inv_condition(geom, q, X):
    """Inverse condition number 1/kappa(J) = s_min / s_max, in [0, 1].
    It is 0 at either singularity type and 1 at a perfectly isotropic
    posture, so a single lower bound on it keeps the robot away from BOTH
    singularity families -- the basis of the certified operating region."""
    s_max, s_min = singular_values(jacobian(geom, q, X))
    return 0.0 if s_max == 0.0 else s_min / s_max


def manipulability(geom, q, X):
    """Yoshikawa's manipulability measure w = |det J| = |det B / det A|."""
    return abs(det_B(geom, q, X) / det_A(geom, q, X))


def dls_inverse(J, mu):
    """Damped least-squares (singularity-robust) inverse of J:
        J_dagger = J^T (J J^T + mu^2 I)^{-1}.
    With damping mu = 0 this is the exact inverse; with mu > 0 the gain in
    every direction is capped at 1/(2 mu), so joint commands stay bounded
    even as a singular value of J shrinks.  Used when a velocity command
    must be formed close to a singular posture."""
    if mu == 0.0:
        return np.linalg.inv(J)
    return J.T @ np.linalg.inv(J @ J.T + mu * mu * np.eye(2))


def velocity_ik(geom, q, X, Xdot):
    """Joint rates for a desired end-effector velocity: qdot = B^{-1} A Xdot.
    Exact, and finite except at Type-1 (det B = 0)."""
    return jacobian_inverse(geom, q, X) @ np.asarray(Xdot, float)


def velocity_fk(geom, q, X, qdot):
    """End-effector velocity for given joint rates: Xdot = A^{-1} B qdot.
    Exact, and finite except at Type-2 (det A = 0)."""
    A, B, _ = AB(geom, q, X)
    return np.linalg.solve(A, B @ np.asarray(qdot, float))
