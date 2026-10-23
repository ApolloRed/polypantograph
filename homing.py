# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
homing.py -- start-up routines: figure out which posture the robot is in, and
calibrate encoder offsets.

Two situations:

  * home(): with encoders that read true angles, the robot still powers up in
    one of several possible postures (working/assembly modes).  Given the
    measured joints and a rough idea of where the end-effector is, home()
    decides the branch labels (omega, sigma) once, so the controller can stay
    on that branch afterwards.

  * calibrate_offsets(): with incremental encoders the readings are only
    correct up to a constant per-joint offset.  If you drive the end-effector
    to a few references of known position (e.g. machined hard stops) and
    record the raw readings, calibrate_offsets recovers those offsets by least
    squares on the loop-closure residual.  No forward kinematics is needed, so
    no assembly mode has to be known.
"""

from dataclasses import dataclass
import numpy as np

from .fkine import fk, constraint
from .ikine import working_modes
from .jacobians import det_A
from .robotgeometry import elbows


@dataclass
class HomingResult:
    omega: tuple        # working modes (omega1, omega2)
    sigma: int          # assembly mode
    X: np.ndarray       # end-effector position at homing
    det_A: float        # det A there (its sign equals sigma)


def home(geom, q_meas, X_approx):
    """Identify the branch labels from one measured pose.

    Computes both assembly-mode candidates from the measured joints and keeps
    whichever forward-kinematics solution is closest to the rough known
    position X_approx; then reads off the working modes.  X_approx only needs
    to be accurate to better than the distance between the two candidates.
    """
    candidates = []
    for sigma in (+1, -1):
        try:
            candidates.append((sigma, fk(geom, q_meas, sigma)))
        except ValueError:
            pass
    if not candidates:
        raise ValueError("no forward-kinematics solution at the homing pose")
    X_approx = np.asarray(X_approx, float)
    sigma, X = min(candidates,
                   key=lambda c: np.linalg.norm(c[1] - X_approx))
    omega = working_modes(geom, q_meas, X)
    return HomingResult(omega, sigma, X, det_A(geom, q_meas, X))


@dataclass
class CalibrationResult:
    offsets: np.ndarray     # add these to the raw readings to get true angles
    rms_residual: float     # meters (leg-length scale)
    iterations: int
    converged: bool


def calibrate_offsets(geom, q_raw_samples, X_known_samples,
                      offsets0=(0.0, 0.0), n_max=50, tol=1e-12):
    """Recover constant encoder offsets by Gauss-Newton least squares.

    For each sample, the residual is the scaled loop-closure violation
        r_i = c_i(q_raw + offset, X_known) / l_{2+i},
    whose derivative with respect to offset_i is -(l_i k_i) / l_{2+i}.  The
    two legs decouple, so the normal equations are diagonal and the solve is
    trivial.  One well-chosen sample suffices in principle; two or more
    spread-out references are recommended for robustness to noise.

    Returns the estimated offsets (true = raw + offset), the final RMS
    residual, the iteration count, and whether it converged.
    """
    Q = np.atleast_2d(np.asarray(q_raw_samples, float))
    Xs = np.atleast_2d(np.asarray(X_known_samples, float))
    if len(Q) != len(Xs):
        raise ValueError("number of joint samples and position samples differ")
    offset = np.asarray(offsets0, float).copy()
    ld = geom.distal
    iters = 0
    converged = False
    for iters in range(1, n_max + 1):
        normal = np.zeros(2)        # diagonal of J^T J (per joint)
        rhs = np.zeros(2)           # J^T r (per joint)
        for q_raw, X in zip(Q, Xs):
            q = q_raw + offset
            b = X - elbows(geom, q)
            c = 0.5 * (np.einsum("ij,ij->i", b, b) - ld ** 2)
            k = np.cos(q) * b[:, 1] - np.sin(q) * b[:, 0]
            jac = -(geom.proximal * k) / ld
            r = c / ld
            normal += jac * jac
            rhs += jac * r
        if np.any(normal < 1e-15):
            break
        delta = -rhs / normal
        offset += delta
        if np.max(np.abs(delta)) < tol:
            converged = True
            break
    # final RMS residual
    ssr = 0.0
    for q_raw, X in zip(Q, Xs):
        b = X - elbows(geom, q_raw + offset)
        c = 0.5 * (np.einsum("ij,ij->i", b, b) - ld ** 2)
        ssr += float(np.sum((c / ld) ** 2))
    rms = np.sqrt(ssr / (2 * len(Q)))
    return CalibrationResult(offset, rms, iters, converged)
