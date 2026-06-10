# polypantograph -- kinematics of the planar five-bar (pantograph) robot
# Author: Abolfazl Mohebbi, PhD. -- Associate Professor, Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
"""
realtime_looper.py -- the per-sample control pipeline, and a fixed-rate
runner built on top of it.

The core is TrackingController.step(X_desired, V_desired): ONE control tick.
It projects the reference onto the safe set, solves inverse kinematics from
the previous sample as a warm start, writes the joint command, optionally
adds a velocity feed-forward, reads the joints back, estimates the executed
pose, and records monitors.  Crucially, the desired point is passed IN each
tick, so it may be produced live (admittance, impedance, teleoperation) -- no
trajectory has to exist ahead of time.

run(trajectory, duration) is a convenience wrapper that simply calls step()
at a fixed rate for callers that DO have the reference in advance.  Its timing
is soft real-time (it paces with the wall clock); for hard timing, call step()
yourself from your own real-time loop.
"""

from dataclasses import dataclass
import time
import numpy as np

from robotio import RobotIO
from jacobians import det_A, inv_condition, velocity_ik
from fkine import FKNewtonEstimator


@dataclass
class StepResult:
    """Everything one control tick produced (also stored in the log)."""
    t: float
    X_des: np.ndarray            # raw desired position
    X_cmd: np.ndarray            # desired after projection onto the safe set
    q_cmd: np.ndarray            # commanded joint angles
    qdot_cmd: np.ndarray | None  # commanded joint rates (if feed-forward on)
    q_meas: np.ndarray           # measured joint angles
    X_exec: np.ndarray           # executed end-effector pose (FK estimate)
    err: float                   # |X_exec - X_cmd|
    iterations: int              # solver loop passes
    flag: bool                   # solver guard tripped / budget exhausted
    det_A: float                 # monitor: sign = assembly mode
    inv_cond: float              # monitor: 1/kappa(J)


@dataclass
class LoopLog:
    """All StepResults from a run, stacked into arrays for analysis/plotting."""
    t: np.ndarray
    X_des: np.ndarray
    X_cmd: np.ndarray
    q_cmd: np.ndarray
    q_meas: np.ndarray
    X_exec: np.ndarray
    err: np.ndarray
    iterations: np.ndarray
    flags: np.ndarray
    det_A: np.ndarray
    inv_cond: np.ndarray

    def summary(self):
        """One-line numeric summary of the run."""
        return ("samples=%d  max|err|=%.3e m  mean iters=%.2f  flags=%d  "
                "min detA=%.3e  min 1/k=%.3f"
                % (len(self.t), self.err.max(), self.iterations.mean(),
                   int(self.flags.sum()), self.det_A.min(),
                   self.inv_cond.min()))


class TrackingController:
    """Runs the inverse-kinematics pipeline one sample at a time.

    Parameters
    ----------
    robot : RobotIO          your hardware wrapper, or a SimulatedRobot.
    geom : FiveBarGeometry
    solver : an IK solver from solvers.py (anything with .solve(X, q)).
    projection : a safe-set projection (BoxWall / CertifiedRegion / chained)
        or None.  Applied to the reference before solving.
    rate_hz : control rate; sets the time step dt = 1 / rate_hz.
    sigma0 : assembly mode from homing; seeds the executed-pose estimator.
    feedforward : if True and a desired velocity is given, also command joint
        rates (written to the robot only if it offers write_dq).

    State carried between ticks: the warm start, the pose estimator, the
    sample counter, and the recorded results (call log() to collect them).
    """

    def __init__(self, robot, geom, solver, projection=None, rate_hz=500.0,
                 sigma0=+1, feedforward=True, record=True):
        self.robot = robot
        self.geom = geom
        self.solver = solver
        self.projection = projection
        self.dt = 1.0 / rate_hz
        self.sigma0 = sigma0
        self.feedforward = feedforward
        self.record = record
        self.estimator = FKNewtonEstimator(geom, sigma0)
        self.reset()

    def reset(self):
        """Re-arm: take the warm start from the robot's current pose and clear
        the recorded history."""
        self._q_star = self.robot.read_q()
        self.estimator.reset(self._q_star)
        self._k = 0
        self._records = []

    def step(self, X_des, V_des=None, advance_sim=True):
        """Execute ONE control tick and return its StepResult.

        Call this from your own real-time loop, passing the desired position
        (and optionally velocity) computed for this sample by whatever runs
        upstream.  advance_sim steps a SimulatedRobot by one period; set
        it False if you advance the simulator yourself.
        """
        t = self._k * self.dt
        X_des = np.asarray(X_des, float)
        # 1) keep the reference in the safe set
        if self.projection is not None:
            X_cmd = self.projection.project(X_des)
        else:
            X_cmd = X_des.copy()
        # 2) inverse kinematics, warm-started from the previous command
        result = self.solver.solve(X_cmd, self._q_star)
        q_cmd = result.q
        self.robot.write_q(q_cmd)
        # 3) optional velocity feed-forward
        qdot_cmd = None
        if self.feedforward and V_des is not None:
            V = np.asarray(V_des, float)
            if self.projection is not None and                     hasattr(self.projection, "tangentialize"):
                V = self.projection.tangentialize(X_cmd, V)
            qdot_cmd = velocity_ik(self.geom, q_cmd, X_cmd, V)
            if hasattr(self.robot, "write_dq"):
                self.robot.write_dq(qdot_cmd)
        # 4) advance a simulator if present, then read the joints back
        if advance_sim and hasattr(self.robot, "step"):
            self.robot.step(self.dt)
        q_meas = self.robot.read_q()
        # 5) estimate the executed pose and the monitors
        estimate = self.estimator.update(q_meas)
        out = StepResult(
            t=t, X_des=X_des, X_cmd=X_cmd, q_cmd=q_cmd, qdot_cmd=qdot_cmd,
            q_meas=q_meas, X_exec=estimate.X,
            err=float(np.linalg.norm(estimate.X - X_cmd)),
            iterations=result.iterations, flag=result.flag,
            det_A=det_A(self.geom, q_cmd, X_cmd),
            inv_cond=inv_condition(self.geom, q_cmd, X_cmd))
        if self.record:
            self._records.append(out)
        self._q_star = q_cmd
        self._k += 1
        return out

    def log(self):
        """Collect everything recorded so far into a LoopLog of arrays."""
        R = self._records
        if not R:
            raise RuntimeError("nothing has been recorded yet")
        return LoopLog(
            t=np.array([r.t for r in R]),
            X_des=np.array([r.X_des for r in R]),
            X_cmd=np.array([r.X_cmd for r in R]),
            q_cmd=np.array([r.q_cmd for r in R]),
            q_meas=np.array([r.q_meas for r in R]),
            X_exec=np.array([r.X_exec for r in R]),
            err=np.array([r.err for r in R]),
            iterations=np.array([r.iterations for r in R]),
            flags=np.array([r.flag for r in R]),
            det_A=np.array([r.det_A for r in R]),
            inv_cond=np.array([r.inv_cond for r in R]))

    def run(self, trajectory, duration, realtime=False):
        """Convenience: drive a pre-planned trajectory (a function
        t -> (X, Xdot)) for duration seconds by calling step() each
        period.  Returns the LoopLog."""
        self.reset()
        n = int(round(duration / self.dt))
        t_next = time.perf_counter()
        for k in range(n):
            X_des, V_des = trajectory(k * self.dt)
            self.step(X_des, V_des)
            if realtime:
                t_next += self.dt
                remaining = t_next - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
        return self.log()
