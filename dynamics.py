# polypantograph -- kinematics and dynamics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
dynamics.py -- the minimal two-coordinate dynamic model of the five-bar, with
inverse and forward dynamics in one place.

The model is the reduced (minimal-coordinate) form

    M(q) qddot + C(q, qdot) qdot + g(q) = tau + Je(q)^T w,

obtained from a constrained Lagrangian by eliminating the passive joints
through the loop-closure constraint.  The two actuated motor angles q are the
generalized coordinates; everything else (elbows, distal-link angles, the
end-effector) is a function of q through the kinematics of the other modules.

Conventions match the rest of the library (leg-grouped numbering): leg i has a
proximal link of length geom.proximal[i] driven by motor i, and a distal link
of length geom.distal[i] from the elbow to the end-effector.  Each moving link
carries a mass, a centre-of-mass offset measured from its inboard joint, and a
rotational inertia about its own centre of mass.  An optional point mass at the
apex models a payload or interaction handle.

Mountings are selected by the in-plane gravity vector passed at construction:
(0, 0) is the horizontal (table-top) mounting, in which gravity does no work
and g(q) vanishes; (0, -9.81) is the vertical (upright) mounting.

The external wrench w may be a planar force [Fx, Fy] or a full [Fx, Fy, Mz];
the moment enters through the orientation of the distal link the tool is
mounted on (tool_leg), via the 3x2 end-effector Jacobian Je = [J; h^T].
"""

from dataclasses import dataclass
import numpy as np

from robotgeometry import elbows
from fkine import fk
from jacobians import jacobian


def _pair(x):
    """Coerce a scalar or length-2 input to a float array of length 2."""
    a = np.atleast_1d(np.asarray(x, float))
    return np.array([a[0], a[0]]) if a.size == 1 else a[:2].copy()


@dataclass
class DynamicParameters:
    """Per-link inertial parameters, in the library's leg-grouped order.

    Each array has the two legs in entries [leg1, leg2].

    Attributes
    ----------
    proximal_mass, distal_mass : (2,) arrays
        Link masses (kg).
    proximal_com : (2,) array
        Centre-of-mass distance of each proximal link from its motor pivot.
    distal_com : (2,) array
        Centre-of-mass distance of each distal link from its elbow joint.
    proximal_inertia, distal_inertia : (2,) arrays
        Rotational inertia of each link about its own centre of mass (the
        out-of-plane component, kg m^2).
    apex_mass : float
        Point mass at the end-effector (kg).
    """
    proximal_mass: np.ndarray
    proximal_com: np.ndarray
    proximal_inertia: np.ndarray
    distal_mass: np.ndarray
    distal_com: np.ndarray
    distal_inertia: np.ndarray
    apex_mass: float = 0.0

    @classmethod
    def uniform(cls, geom, proximal_mass, distal_mass, apex_mass=0.0):
        """Build parameters for uniform slender rods: the centre of mass is at
        each link's midpoint and the inertia is m L^2 / 12.  Masses may be a
        scalar (same for both legs) or a length-2 pair."""
        pm = _pair(proximal_mass)
        dm = _pair(distal_mass)
        lp = geom.proximal
        ld = geom.distal
        return cls(proximal_mass=pm, proximal_com=lp / 2.0,
                   proximal_inertia=pm * lp ** 2 / 12.0,
                   distal_mass=dm, distal_com=ld / 2.0,
                   distal_inertia=dm * ld ** 2 / 12.0,
                   apex_mass=float(apex_mass))

    def __post_init__(self):
        # normalise everything to length-2 float arrays
        for f in ("proximal_mass", "proximal_com", "proximal_inertia",
                  "distal_mass", "distal_com", "distal_inertia"):
            object.__setattr__(self, f, _pair(getattr(self, f)))
        object.__setattr__(self, "apex_mass", float(self.apex_mass))


class FiveBarDynamics:
    """Minimal-coordinate dynamics of the five-bar.

    Parameters
    ----------
    geom : FiveBarGeometry
    params : DynamicParameters
    gravity : pair (gx, gy)
        In-plane gravitational acceleration (m/s^2).  (0, 0) is horizontal
        (g(q) = 0); (0, -9.81) is vertical.
    sigma : +1 or -1
        Assembly mode used by the forward kinematics (carried from homing).
    tool_leg : 0 or 1
        Distal link the end-effector tool is rigidly mounted on; sets which
        passive-rate row carries an applied moment Mz.
    """

    def __init__(self, geom, params, gravity=(0.0, 0.0), sigma=+1, tool_leg=0):
        self.geom = geom
        self.params = params
        self.gravity = np.asarray(gravity, float)
        self.sigma = sigma
        self.tool_leg = int(tool_leg)

    # --- kinematic maps reused by every dynamic quantity ----------------
    def _maps(self, q):
        """Return the velocity maps at q: end-effector position X, position
        Jacobian J (Xdot = J qdot), elbow-velocity maps GE[i], distal
        centre-of-mass maps Gc[i], and passive-rate rows h[i]
        (phidot_i = h[i] . qdot)."""
        geom, p = self.geom, self.params
        q = np.asarray(q, float)
        X = fk(geom, q, self.sigma)
        J = jacobian(geom, q, X)
        s, c = np.sin(q), np.cos(q)
        GE = [np.zeros((2, 2)), np.zeros((2, 2))]
        GE[0][:, 0] = geom.proximal[0] * np.array([-s[0], c[0]])
        GE[1][:, 1] = geom.proximal[1] * np.array([-s[1], c[1]])
        b = X - elbows(geom, q)                 # rows b_i = X - E_i
        phi = np.arctan2(b[:, 1], b[:, 0])
        ld = geom.distal
        h = [np.zeros(2), np.zeros(2)]
        Gc = [None, None]
        for i in range(2):
            n_i = np.array([-np.sin(phi[i]), np.cos(phi[i])])
            h[i] = (n_i @ (J - GE[i])) / ld[i]
            Gc[i] = GE[i] + (p.distal_com[i] / ld[i]) * (J - GE[i])
        return {"X": X, "J": J, "GE": GE, "Gc": Gc, "h": h, "phi": phi}

    # --- the three configuration-dependent quantities -------------------
    def mass_matrix(self, q):
        """Symmetric positive-definite mass matrix M(q) (2x2)."""
        p = self.params
        m = self._maps(q)
        M = np.zeros((2, 2))
        for i in range(2):                      # proximal links (about pivot)
            M[i, i] += p.proximal_inertia[i] + p.proximal_mass[i]                 * p.proximal_com[i] ** 2
        for i in range(2):                      # distal links
            M += p.distal_mass[i] * (m["Gc"][i].T @ m["Gc"][i])                 + p.distal_inertia[i] * np.outer(m["h"][i], m["h"][i])
        M += p.apex_mass * (m["J"].T @ m["J"])  # apex point mass
        return 0.5 * (M + M.T)

    def coriolis(self, q, qd, fd_step=1e-6):
        """Coriolis/centrifugal matrix C(q, qdot) from the Christoffel symbols
        of M, with the mass-matrix partials taken by central difference.  This
        choice keeps Mdot - 2C skew-symmetric to machine precision."""
        q = np.asarray(q, float)
        qd = np.asarray(qd, float)
        dM = []
        for k in range(2):
            dq = np.zeros(2); dq[k] = fd_step
            dM.append((self.mass_matrix(q + dq) - self.mass_matrix(q - dq))
                      / (2 * fd_step))
        C = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                C[i, j] = sum(0.5 * (dM[k][i, j] + dM[j][i, k] - dM[i][j, k])
                              * qd[k] for k in range(2))
        return C

    def gravity_torque(self, q):
        """Gravity vector g(q); zero in the horizontal mounting.

        g(q) = -sum_k m_k Gc_k^T g_vec, the gradient of the potential energy,
        summed over the four links and the apex mass."""
        if not np.any(self.gravity):
            return np.zeros(2)
        geom, p = self.geom, self.params
        m = self._maps(q)
        g = np.zeros(2)
        for i in range(2):                      # proximal COM maps
            Gcp = (p.proximal_com[i] / geom.proximal[i]) * m["GE"][i]
            g += -p.proximal_mass[i] * (Gcp.T @ self.gravity)
        for i in range(2):                      # distal COM maps
            g += -p.distal_mass[i] * (m["Gc"][i].T @ self.gravity)
        g += -p.apex_mass * (m["J"].T @ self.gravity)
        return g

    # --- end-effector Jacobian and wrench map ---------------------------
    def end_effector_jacobian(self, q):
        """3x2 end-effector Jacobian Je = [J; h_tool^T] mapping qdot to the
        twist [xdot, ydot, theta_e dot], with theta_e the orientation of the
        distal link the tool is mounted on (tool_leg)."""
        m = self._maps(q)
        return np.vstack([m["J"], m["h"][self.tool_leg]])

    def _ext_genforce(self, q, wrench):
        """Generalized joint force from an apex wrench: J^T F (+ Mz h) when a
        moment is supplied.  wrench is None, [Fx,Fy], or [Fx,Fy,Mz]."""
        if wrench is None:
            return np.zeros(2)
        w = np.asarray(wrench, float)
        m = self._maps(q)
        f = m["J"].T @ w[:2]
        if w.size >= 3:
            f = f + w[2] * m["h"][self.tool_leg]
        return f

    # --- the two dynamics primitives ------------------------------------
    def inverse_dynamics(self, q, qd, qdd, wrench=None):
        """Torque tau that produces the desired acceleration qdd at (q, qdot),
        optionally balancing an apex wrench (Algorithm: inverse dynamics)."""
        q = np.asarray(q, float)
        return (self.mass_matrix(q) @ np.asarray(qdd, float)
                + self.coriolis(q, qd) @ np.asarray(qd, float)
                + self.gravity_torque(q) - self._ext_genforce(q, wrench))

    def forward_dynamics(self, q, qd, tau, wrench=None):
        """Acceleration qddot produced by the torque tau at (q, qdot),
        optionally with an apex wrench (Algorithm: forward dynamics)."""
        q = np.asarray(q, float)
        qd = np.asarray(qd, float)
        rhs = (np.asarray(tau, float) + self._ext_genforce(q, wrench)
               - self.coriolis(q, qd) @ qd - self.gravity_torque(q))
        return np.linalg.solve(self.mass_matrix(q), rhs)

    def step(self, q, qd, tau, dt, wrench=None):
        """Advance the state one time step by semi-implicit (symplectic)
        Euler: qddot = FD(...), then qdot += dt qddot, q += dt qdot.  Returns
        (q_next, qdot_next).  For long horizons integrate with a higher-order
        scheme in the caller using forward_dynamics directly."""
        qdd = self.forward_dynamics(q, qd, tau, wrench)
        qd_next = np.asarray(qd, float) + dt * qdd
        q_next = np.asarray(q, float) + dt * qd_next
        return q_next, qd_next

    # --- energies (useful for checks) -----------------------------------
    def kinetic_energy(self, q, qd):
        """T = 1/2 qdot^T M(q) qdot."""
        qd = np.asarray(qd, float)
        return 0.5 * qd @ self.mass_matrix(q) @ qd

    def potential_energy(self, q):
        """Gravitational potential energy (0 in the horizontal mounting)."""
        if not np.any(self.gravity):
            return 0.0
        geom, p = self.geom, self.params
        q = np.asarray(q, float)
        E = elbows(geom, q)
        X = fk(geom, q, self.sigma)
        ahat = np.stack([np.cos(q), np.sin(q)], axis=1)
        V = 0.0
        for i in range(2):
            rcp = geom.bases[i] + p.proximal_com[i] * ahat[i]
            V += -p.proximal_mass[i] * (self.gravity @ rcp)
        for i in range(2):
            u = X - E[i]
            u = u / np.linalg.norm(u)
            rcd = E[i] + p.distal_com[i] * u
            V += -p.distal_mass[i] * (self.gravity @ rcd)
        V += -p.apex_mass * (self.gravity @ X)
        return float(V)
