import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))
import numpy as np
from robotgeometry import FiveBarGeometry
from ikine import ik
from simrobot import SimulatedRobot
from ikine import PerLegNewtonSolver
from realtime_looper import TrackingController
from virtualwall import BoxWall
import trajectorygen as tg

G = FiveBarGeometry.symmetric(0.20, 0.25, 0.20)
CENTER = (0.0, 0.30)


def _controller(robot, projection=None):
    return TrackingController(robot, G, PerLegNewtonSolver(G),
                              projection=projection, rate_hz=500)


def test_tracking_clean_circle():
    robot = SimulatedRobot(G, ik(G, [0.08, 0.30], (+1, -1)))
    log = _controller(robot).run(tg.circle(CENTER, 0.08, 2.0), 2.0)
    assert log.err.max() < 1e-9
    assert log.flags.sum() == 0
    assert log.det_A.min() > 0
    assert log.inv_cond.min() > 0.31     # true min on this path: 0.317


def test_tracking_with_quantized_encoders():
    robot = SimulatedRobot(G, ik(G, [0.08, 0.30], (+1, -1)),
                           counts_per_rev=2 ** 16)
    log = _controller(robot).run(tg.circle(CENTER, 0.08, 2.0), 2.0)
    assert log.err.max() < 0.5e-3
    assert log.flags.sum() == 0


def test_wall_guarantee():
    wall = BoxWall(-0.12, 0.06, 0.20, 0.40)
    robot = SimulatedRobot(G, ik(G, [0.06, 0.30], (+1, -1)))
    log = _controller(robot, wall).run(tg.circle(CENTER, 0.08, 2.0), 2.0)
    assert log.X_cmd[:, 0].max() <= 0.06 + 1e-12
    assert log.X_exec[:, 0].max() <= 0.06 + 1e-6
    assert log.err.max() < 1e-9


def test_streaming_step_matches_run():
    traj = tg.circle(CENTER, 0.08, 2.0)
    q0 = ik(G, [0.08, 0.30], (+1, -1))
    log_run = _controller(SimulatedRobot(G, q0)).run(traj, 1.0)
    ctrl = _controller(SimulatedRobot(G, q0))
    for k in range(int(1.0 * 500)):
        X, V = traj(k * ctrl.dt)
        ctrl.step(X, V)
    log_step = ctrl.log()
    assert np.allclose(log_run.q_cmd, log_step.q_cmd, atol=1e-12)
    assert np.allclose(log_run.err, log_step.err, atol=1e-12)


def test_streaming_command_depends_on_state():
    ctrl = _controller(SimulatedRobot(G, ik(G, [0.0, 0.30], (+1, -1))))
    X_d = np.array([0.0, 0.30])
    for k in range(500):
        res = ctrl.step(X_d)
        X_d = res.X_exec + np.array([1e-4, 0.0])
    log = ctrl.log()
    assert log.flags.sum() == 0
    assert log.X_cmd[-1, 0] > 0.04
    assert log.err.max() < 1e-8


def test_trajconcat():
    traj = tg.trajconcat((1.0, tg.hold(CENTER)),
                         (1.0, tg.circle(CENTER, 0.05, 1.0)))
    X0, _ = traj(0.5)
    assert np.allclose(X0, CENTER)
    X1, _ = traj(1.0)             # start of the circle segment
    assert np.allclose(X1, np.array(CENTER) + [0.05, 0.0], atol=1e-9)
