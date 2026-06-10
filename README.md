# polypantograph

A small, readable Python library for the **kinematics and dynamics of the
planar five-bar (pantograph) parallel robot**. With general link lengths and
inertias it provides everything needed to model, simulate, and control the
mechanism:

- **Kinematics** — closed-form and Newton forward/inverse kinematics with
  explicit handling of the assembly mode and the two working modes, so you
  always know which branch you are on.
- **Jacobians and singularities** — the `A`/`B` matrices, the end-effector
  Jacobian and its damped (singularity-robust) inverse, determinants,
  conditioning, and manipulability.
- **Safety regions** — box keep-in walls and a conditioning-certified region
  to keep motion away from singularities.
- **Soft-real-time loop** — a per-sample tracking controller for
  admittance / impedance / teleoperation, plus a forward-kinematics estimator
  with live monitors.
- **Dynamics** — the minimal two-coordinate rigid-body model (mass matrix,
  Coriolis/centrifugal, gravity) with inverse and forward dynamics, valid for
  horizontal or vertical mounting and for an end-effector force or full wrench.
- **Model-based control building blocks** — computed torque, impedance, and
  hybrid force/position, all built on the dynamics above.
- **Parameters in code or URDF** — read parameters from a URDF, or write one
  from values given in code.

Geometry and inertia need not be symmetric or uniform: the two legs may have
different link lengths, and each link its own mass, centre-of-mass offset, and
inertia. The library is hardware-agnostic — you supply a thin two-motor I/O
class — and ships with a simulated robot, so everything runs and is tested
without hardware. The only hard dependency is NumPy (Matplotlib is used for the
optional plots).

It is the reference software accompanying two papers, which explain the
mathematics that the docstrings here point back to:

- A. Mohebbi, *Kinematic Modeling, Analysis, and Real-Time Implementation of a
  Planar Five-Bar Pantograph Robot*, Zenodo, 2026.
  doi: [10.5281/zenodo.20616265](https://doi.org/10.5281/zenodo.20616265)
- A. Mohebbi, *Dynamic Modeling, Control, and Real-Time Implementation of a
  Planar Five-Bar Pantograph Robot*, Zenodo, 2026.
  doi: [10.5281/zenodo.20620179](https://doi.org/10.5281/zenodo.20620179)

**Author:** Abolfazl Mohebbi, PhD. — Associate Professor, Polytechnique Montréal
**License:** PolyForm Noncommercial License 1.0.0 (free for non-commercial use;
see `LICENSE`).

## Files

All modules sit in one folder (run things from this folder).

| File | What it holds |
|---|---|
| `robotgeometry.py` | `FiveBarGeometry` (the four link lengths + base), `wrap`, `elbows` |
| `fkine.py` | forward kinematics: `fk`, the `FKNewtonEstimator`, plus `distal_angles`, `constraint` |
| `ikine.py` | inverse kinematics: `ik`, `ik_branches`, `working_modes`, and the three IK solvers |
| `jacobians.py` | `A`/`B` matrices, `jacobian`, inverse, determinants, conditioning, damped inverse, velocity maps |
| `dynamics.py` | `FiveBarDynamics` (mass matrix, Coriolis, gravity, inverse/forward dynamics, `step`) and `DynamicParameters` |
| `urdf_loader.py` | `load_urdf` / `build_dynamics` — read geometry and inertial parameters from a URDF |
| `urdf_builder.py` | `write_urdf` / `write_urdf_symmetric` — write a URDF from parameters given in code |
| `virtualwall.py` | `BoxWall` and the conditioning-`CertifiedRegion` keep-in sets |
| `robotio.py` | `RobotIO`, the interface to your hardware API |
| `simrobot.py` | `SimulatedRobot` for testing without hardware |
| `homing.py` | start-up branch identification and encoder-offset calibration |
| `trajectorygen.py` | circle / line / hold trajectory generators and `trajconcat` |
| `realtime_looper.py` | `TrackingController` — the per-sample control pipeline |
| `plotting.py` | optional figures from a run (needs Matplotlib) |
| `tests/` | the test suite (`python -m pytest tests`) |
| `examples/` | runnable examples and two sample URDFs — `fivebar.urdf` (symmetric) and `general_arm.urdf` (asymmetric) |

## Install

There is no packaging step — get the dependencies and run from this folder:

```
pip install numpy matplotlib pytest
python -m pytest tests          # run the test suite
python examples/example_IK.py   # run an example
```

NumPy is the only requirement for the core library; Matplotlib is needed for
the plots and most examples, and pytest to run the tests.

## Quick start (simulated robot)

**A symmetric, uniform robot — kinematic tracking.** Drive a simulated robot
around a circle through the per-sample tracking controller:

```python
from robotgeometry import FiveBarGeometry
from ikine import ik, PerLegNewtonSolver
from simrobot import SimulatedRobot
from realtime_looper import TrackingController
import trajectorygen as tg

geom = FiveBarGeometry.symmetric(a=0.20, b=0.25, d=0.20)   # or (l1,l2,l3,l4,d)
robot = SimulatedRobot(geom, ik(geom, [0.08, 0.30], omega=(+1, -1)),
                       counts_per_rev=2**16)
ctrl = TrackingController(robot, geom, PerLegNewtonSolver(geom), rate_hz=500)
log = ctrl.run(tg.circle((0.0, 0.30), 0.08, period=3.0), duration=3.0)
print(log.summary())
```

**An asymmetric, non-uniform robot — dynamic simulation.** The legs have
different link lengths, and each link its own mass, centre-of-mass offset, and
inertia. Build the dynamics and simulate with the forward-dynamics step:

```python
import numpy as np
from robotgeometry import FiveBarGeometry
from dynamics import DynamicParameters, FiveBarDynamics
from ikine import ik
from fkine import fk

# asymmetric geometry (legs differ) + non-uniform, per-link inertia
geom = FiveBarGeometry(l1=0.18, l2=0.22, l3=0.23, l4=0.27, d=0.19)
params = DynamicParameters(
    proximal_mass=[0.45, 0.55], proximal_com=[0.07, 0.12],
    proximal_inertia=[0.0012, 0.0020],
    distal_mass=[0.28, 0.34], distal_com=[0.10, 0.16],
    distal_inertia=[0.0011, 0.0019], apex_mass=0.08)
dyn = FiveBarDynamics(geom, params, gravity=(0.0, -9.81))   # vertical mounting

q = ik(geom, [0.0, 0.30], omega=(+1, -1)); qd = np.zeros(2)
g = dyn.gravity_torque(q)                       # gravity load at this pose
qdd = dyn.forward_dynamics(q, qd, tau=g)        # ~0: this torque balances gravity
for _ in range(1000):                           # simulate 1 s at 1 kHz
    q, qd = dyn.step(q, qd, g, dt=1e-3)
print("held near start:", fk(geom, q))
```

A symmetric, uniform robot is the easy special case of the same code:
`FiveBarGeometry.symmetric(a, b, d)` and
`DynamicParameters.uniform(geom, proximal_mass, distal_mass, apex_mass)`.

## Per-sample use (admittance / impedance / teleoperation)

Nothing assumes the command exists in advance. The basic unit is one tick,
`ctrl.step(X_desired, V_desired)`, which you call from your own loop with the
command you computed for this sample:

```python
while running:
    F = read_force_sensor()             # your sensor
    X_d, V_d = admittance.update(F)     # command for THIS sample
    res = ctrl.step(X_d, V_d)           # one inverse-kinematics tick
    # res.X_exec, res.inv_cond, res.det_A are available now, in time for
    # the next sample's decision
log = ctrl.log()
```

`run(trajectory, duration)` is just a wrapper that calls `step` at a fixed rate.
On the forward side, `FKNewtonEstimator` (in `fkine.py`) `.update(q_meas,
qdot_meas)` is likewise one call per sample and returns the pose, velocity, and
live monitors (`det_A`, `inv_cond`). See `examples/example_IK.py` and
`examples/example_FK.py` (each also shows, in comments, how to plug in a real
robot through `robotio.py`).

## Dynamics

`dynamics.py` provides the minimal two-coordinate model
`M(q) q̈ + C(q,q̇) q̇ + g(q) = τ + Jₑ(q)ᵀ w`, built on the same kinematic maps as
the rest of the library. The mounting is set by the in-plane gravity vector:
`(0, 0)` is horizontal (gravity does no work, `g(q)=0`), `(0, -9.81)` is
vertical. An apex point mass models a payload, and an applied moment `Mz`
enters through the end-effector orientation Jacobian.

```python
from urdf_loader import build_dynamics
dyn, geom = build_dynamics("examples/fivebar.urdf", gravity=(0.0, -9.81))
tau = dyn.inverse_dynamics(q, qd, qdd)          # feedforward torque
qdd = dyn.forward_dynamics(q, qd, tau)          # simulation
q, qd = dyn.step(q, qd, tau, dt)                # one symplectic-Euler step
```

Parameters can be set in code with `DynamicParameters.uniform(geom,
proximal_mass, distal_mass, apex_mass)` (uniform slender rods) or by giving the
per-link masses, centre-of-mass offsets, and inertias explicitly, as in the
quick start above. The examples `example_dynamics.py` (computed torque),
`example_impedance.py` (impedance control with a disturbance), and
`example_general_robot.py` (an asymmetric, non-uniform robot) show the model in
closed loop; `tests/test_dynamics.py`, `tests/test_dynamics_general.py`, and
`tests/test_impedance.py` exercise it.

**URDF.** URDF cannot encode a closed loop, so the five-bar is written as a
spanning tree: two serial 2R legs off a common base, each ending in a fixed tip
at the apex. The loop closure is supplied by the model, not the file; the URDF
only carries link lengths (from joint origins) and inertial parameters. See
`examples/fivebar.urdf` and the convention documented in `urdf_loader.py`. If
you do not have a URDF, `urdf_builder.py` writes one from parameters given in
code (`write_urdf(path, geom, params, name=...)`, or `write_urdf_symmetric`
for a symmetric uniform-rod robot from scalars); the file it writes loads back
through `load_urdf` to the same parameters.

## License

Copyright (c) 2026 Abolfazl Mohebbi.

This software is licensed under the PolyForm Noncommercial License 1.0.0.

You may use, copy, modify, and share this software for non-commercial purposes only.

Commercial use is not permitted without separate written permission from the author.

For commercial licensing, please contact: abolfazl.mohebbi@polymtl.ca

## How to cite

If this library or the underlying methods are useful in your work, please cite
the papers:

```bibtex
@misc{mohebbi_2026_20616265,
  author    = {Mohebbi, Abolfazl},
  title     = {Kinematic Modeling, Analysis, and Real-Time Implementation of a Planar Five-Bar Pantograph Robot},
  month     = jun,
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20616265},
  url       = {https://doi.org/10.5281/zenodo.20616265}
}

@misc{mohebbi_2026_20620179,
  author    = {Mohebbi, Abolfazl},
  title     = {Dynamic Modeling, Control, and Real-Time Implementation of a Planar Five-Bar Pantograph Robot},
  month     = jun,
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20620179},
  url       = {https://doi.org/10.5281/zenodo.20620179}
}
```

## Conventions

Angles in radians, lengths in meters; joint angles measured CCW from +x;
motors at (±d/2, 0). Solver tolerances are in meters at the leg-length scale
(`sum_i |c_i| / l_{2+i} <= eps`). The control loop is *soft* real time on a
normal OS; for hard timing, call `step()` from your own real-time scheduler —
every piece works standalone.