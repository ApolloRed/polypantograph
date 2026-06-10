# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
virtualwall.py -- ways to keep the commanded end-effector inside a safe set.

A "projection" object answers two questions: project(X) returns the nearest
allowed point, and contains(X) says whether a point is allowed.  Projections
are applied to the REFERENCE, before the inverse-kinematics solve, so the
safety guarantee holds no matter which solver runs underneath.

Two kinds are provided:
  * BoxWall          - a simple rectangular keep-in box (axis-aligned virtual
                       walls), with an optional velocity tangentialization so
                       a feed-forward command does not push into a wall.
  * CertifiedRegion  - the conditioning-certified operating region of the
                       region: a grid of end-effector points that are reachable
                       with margin, in the right branches, and well away from
                       BOTH singularity types (1/kappa >= a chosen floor).
"""

import numpy as np


class BoxWall:
    """Axis-aligned rectangular keep-in region [xmin, xmax] x [ymin, ymax]."""

    def __init__(self, xmin, xmax, ymin, ymax):
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("empty box (need xmin < xmax and ymin < ymax)")
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

    def project(self, X):
        """Clamp X into the box (nearest point inside)."""
        X = np.asarray(X, float)
        return np.array([np.clip(X[0], self.xmin, self.xmax),
                         np.clip(X[1], self.ymin, self.ymax)])

    def contains(self, X, tol=0.0):
        """True if X lies in the box (within an optional tolerance)."""
        X = np.asarray(X, float)
        return (self.xmin - tol <= X[0] <= self.xmax + tol and
                self.ymin - tol <= X[1] <= self.ymax + tol)

    def tangentialize(self, X, Xdot):
        """Remove the part of a velocity that points into an active wall, so
        the feed-forward slides along the wall instead of through it."""
        X = np.asarray(X, float)
        V = np.asarray(Xdot, float).copy()
        if (X[0] <= self.xmin and V[0] < 0) or (X[0] >= self.xmax and V[0] > 0):
            V[0] = 0.0
        if (X[1] <= self.ymin and V[1] < 0) or (X[1] >= self.ymax and V[1] > 0):
            V[1] = 0.0
        return V


class CertifiedRegion:
    """The conditioning-certified operating region.

    Built once by scanning a grid of candidate end-effector points and keeping
    only those that, in the homed working/assembly modes, are: reachable by
    both legs with a margin, consistent (IK then FK returns the same point),
    in the correct assembly mode, and well-conditioned (1/kappa(J) >= kappa0).
    Because 1/kappa goes to zero at BOTH singularity types, the single floor
    kappa0 simultaneously keeps a margin from the reach boundary (Type-1) and
    from the interior collinear locus (Type-2).

    project(X) returns the nearest certified grid point; contains(X) is True
    when X is within about one grid cell of the region.  Building costs one
    pass over the grid (vectorized, a second or two at a few-millimeter cell).
    """

    def __init__(self, points, kappa0, cell):
        self.points = np.asarray(points, float)
        self.kappa0 = kappa0
        self.cell = cell
        if len(self.points) == 0:
            raise ValueError("certified region is empty; relax kappa0 or the "
                             "margins, or widen the bounds")

    @classmethod
    def build(cls, geom, omega=(+1, -1), sigma=+1, kappa0=0.4,
              reach_margin=0.005, bounds=None, cell=0.002):
        """Scan a grid and return the certified region.

        Parameters
        ----------
        geom : FiveBarGeometry
        omega, sigma : the homed working and assembly modes to certify in.
        kappa0 : the conditioning floor (1/kappa lower bound).
        reach_margin : keep at least this far (meters) inside each annulus.
        bounds : (xmin, xmax, ymin, ymax); a sensible default is derived from
            the link lengths if omitted.
        cell : grid spacing in meters.
        """
        if bounds is None:
            reach = max(geom.l1 + geom.l3, geom.l2 + geom.l4)
            bounds = (-geom.d / 2 - reach, geom.d / 2 + reach, -reach, reach)
        xs = np.arange(bounds[0], bounds[1] + cell / 2, cell)
        ys = np.arange(bounds[2], bounds[3] + cell / 2, cell)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

        # 1) reach margins on both annuli
        ok = np.ones(len(pts), bool)
        for i in range(2):
            r = np.linalg.norm(pts - geom.bases[i], axis=1)
            rmin, rmax = geom.reach_annulus(i)
            ok &= (r >= rmin + reach_margin) & (r <= rmax - reach_margin)

        # 2) inverse kinematics in the homed working modes (vectorized)
        idx = np.where(ok)[0]
        P = pts[idx]
        q = np.empty((len(P), 2))
        good = np.ones(len(P), bool)
        for i in range(2):
            L = P - geom.bases[i]
            r = np.linalg.norm(L, axis=1)
            lp, ld = geom.proximal[i], geom.distal[i]
            cb = (r * r + lp * lp - ld * ld) / (2.0 * lp * r)
            good &= (cb >= -1.0) & (cb <= 1.0)
            q[:, i] = (np.arctan2(L[:, 1], L[:, 0])
                       + omega[i] * np.arccos(np.clip(cb, -1.0, 1.0)))

        # 3) assembly mode and conditioning, from the constraint matrices
        E = np.empty((len(P), 2, 2))
        for i in range(2):
            E[:, i, 0] = geom.bases[i, 0] + geom.proximal[i] * np.cos(q[:, i])
            E[:, i, 1] = geom.bases[i, 1] + geom.proximal[i] * np.sin(q[:, i])
        b = P[:, None, :] - E                                  # (N,2,2)
        detA = b[:, 0, 0] * b[:, 1, 1] - b[:, 0, 1] * b[:, 1, 0]
        good &= (np.sign(detA) == sigma) & (np.abs(detA) > 1e-12)
        k = np.cos(q) * b[:, :, 1] - np.sin(q) * b[:, :, 0]    # (N,2)
        Bdiag = geom.proximal[None, :] * k
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.empty_like(b)                             # A^{-1}, per point
            inv[:, 0, 0] = b[:, 1, 1] / detA
            inv[:, 0, 1] = -b[:, 0, 1] / detA
            inv[:, 1, 0] = -b[:, 1, 0] / detA
            inv[:, 1, 1] = b[:, 0, 0] / detA
            J = inv * Bdiag[:, None, :]                        # J = A^{-1} B
            g11 = J[:, 0, 0] ** 2 + J[:, 1, 0] ** 2
            g22 = J[:, 0, 1] ** 2 + J[:, 1, 1] ** 2
            trace = g11 + g22
            det = np.abs(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
            disc = np.sqrt(np.maximum(trace * trace - 4 * det * det, 0.0))
            s_max = np.sqrt(np.maximum((trace + disc) / 2, 0.0))
            s_min = np.sqrt(np.maximum((trace - disc) / 2, 0.0))
            inv_cond = np.where(s_max > 0, s_min / s_max, 0.0)
        good &= inv_cond >= kappa0
        return cls(P[good], kappa0, cell)

    def project(self, X):
        """Nearest certified point to X."""
        X = np.asarray(X, float)
        diff = self.points - X
        i = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))
        return self.points[i].copy()

    def contains(self, X, tol=None):
        """True if X is within about one grid cell of the certified region."""
        tol = 1.5 * self.cell if tol is None else tol
        return bool(np.linalg.norm(self.project(X) - np.asarray(X, float))
                    <= tol)


class ChainedProjection:
    """Apply several projections one after another (e.g. certified region,
    then a tighter box).  For the common region-then-box use this is a good,
    cheap approximation to projecting onto the intersection."""

    def __init__(self, *projections):
        self.projections = projections

    def project(self, X):
        for p in self.projections:
            X = p.project(X)
        return np.asarray(X, float)

    def contains(self, X):
        return all(p.contains(X) for p in self.projections)
