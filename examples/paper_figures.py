# polypantograph example -- Author: Abolfazl Mohebbi, PhD., Polytechnique Montreal
# License: PolyForm Noncommercial License 1.0.0 (non-commercial use only)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
paper_figures.py -- regenerate the figures (sim1..sim4): simulated robot + control loop 
+ plotting.  Output goes to ./figures/.
"""

import numpy as np
from robotgeometry import FiveBarGeometry
from ikine import ik
from simrobot import SimulatedRobot
from virtualwall import BoxWall, CertifiedRegion
from ikine import ClosedFormSolver, TaskNewtonSolver, PerLegNewtonSolver
from realtime_looper import TrackingController
import trajectorygen as tg
import plotting

geom = FiveBarGeometry.symmetric(a=0.20, b=0.25, d=0.20)
OMEGA, SIGMA, RATE = (+1, -1), +1, 500.0
os.makedirs("figures", exist_ok=True)
plotting.setup_style()


def solvers():
    return [("(a) closed form", ClosedFormSolver(geom, OMEGA)),
            ("(b) task Newton", TaskNewtonSolver(geom, SIGMA)),
            ("(c) per-leg Newton", PerLegNewtonSolver(geom))]


def run(solver, traj, duration, projection=None, **robot_kw):
    X0 = traj(0.0)[0]
    X0 = projection.project(X0) if projection is not None else X0
    robot = SimulatedRobot(geom, ik(geom, X0, OMEGA), **robot_kw)
    ctrl = TrackingController(robot, geom, solver, projection=projection,
                              rate_hz=RATE, sigma0=SIGMA)
    return ctrl.run(traj, duration)


# SIM 1: three solvers, clean circle
traj1 = tg.circle((0.0, 0.30), 0.08, period=3.0)
logs1 = {nm: run(sv, traj1, 3.0) for nm, sv in solvers()}
print("--- SIM1 ---")
for nm, log in logs1.items():
    print("%s: %s" % (nm, log.summary()))
plotting.plot_tracking_comparison(geom, logs1, SIGMA).savefig("figures/sim1.pdf")

# SIM 2: virtual wall, three solvers
wall = BoxWall(-0.12, 0.06, 0.20, 0.40)
logs2 = {nm: run(sv, traj1, 3.0, projection=wall) for nm, sv in solvers()}
print("--- SIM2 ---")
for nm, log in logs2.items():
    print("%s: %s  max x = %.6f" % (nm, log.summary(), log.X_cmd[:, 0].max()))
plotting.plot_wall_comparison(geom, logs2, wall, SIGMA).savefig("figures/sim2.pdf")

# SIM 3: singularity encounters, protection on
print("building certified region (2 mm grid)...")
region = CertifiedRegion.build(geom, OMEGA, SIGMA, kappa0=0.4,
                               reach_margin=0.005,
                               bounds=(-0.34, 0.34, -0.05, 0.46), cell=0.002)
damped = PerLegNewtonSolver(geom, mu=5e-3)
traj3a = tg.circle((0.14, 0.34), 0.10, period=3.0, phase=np.pi)
log3a = run(damped, traj3a, 3.0, projection=region)
traj3b = tg.trajconcat(
    (1.2, tg.line([0.02, 0.30], [0.02, 0.02], 1.2, smooth=False)),
    (1.2, tg.line([0.02, 0.02], [0.02, 0.30], 1.2, smooth=False)))
log3b = run(damped, traj3b, 2.4, projection=region)
print("--- SIM3 ---")
print("Type-1: %s" % log3a.summary())
print("Type-2: %s" % log3b.summary())
plotting.plot_singularity_encounters(geom, log3a, log3b, region.kappa0,
                                     reach_margin=0.005,
                                     omega=OMEGA).savefig("figures/sim3.pdf")

# SIM 4: FK from quantized encoders
CPR = 2 ** 16
log4 = run(PerLegNewtonSolver(geom), traj1, 3.0, counts_per_rev=CPR)
print("--- SIM4 ---")
print(log4.summary())
plotting.plot_fk_quantized(geom, log4, 2 * np.pi / CPR, SIGMA).savefig(
    "figures/sim4.pdf")
print("figures written to ./figures/")
