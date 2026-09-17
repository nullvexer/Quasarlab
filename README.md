# QuasarLab

QuasarLab is a deterministic computational physics laboratory. Its physics
engine is the source of truth: results are produced by explicit computation,
not by heuristics or generated guesses.

This repository contains **Version 0.2.0**: a 2D particle driven by a
composable set of force laws — uniform gravity, constant applied forces,
linear drag, quadratic drag, and Coulomb friction on plane surfaces —
integrated with explicit Euler at a fixed timestep and validated against
exact analytical solutions wherever trustworthy closed forms exist.

## Current capabilities

- 2D particle with mass, position, and velocity in SI units
- Composable force laws: any number of laws act simultaneously
- Uniform gravity `F_g = m g`
- Constant applied force `F = (Fx, Fy)`
- Linear drag `F_d = -b v` (`b >= 0`)
- Quadratic drag `F_d = -(1/2) rho Cd A |v| v`
- Coulomb friction (static + kinetic) on rigid plane surfaces, with contact
  mechanics separated from the friction law
- Engine-side force breakdown: individual contributions, net force, and
  acceleration at any state (`World.evaluate_forces`)
- Explicit Euler numerical integration, fixed timestep
- Trajectory recording with `time`, `position`, and `velocity` arrays
- Exact analytical references: constant acceleration, 1D linear drag,
  1D quadratic-drag fall (speed and distance)
- Numerical-vs-analytical error checks and convergence tests
- Matplotlib trajectory and component-vs-time visualization
- Tkinter graphical interface (`quasarlab_gui.py`) with five scenarios
- GitHub Actions CI and a Windows `.exe` build/smoke-test workflow
- Strict rejection of invalid physical and numerical input

## Physics implemented

Newton's second law with force composition:

```text
F_net = sum(F_i)
a     = F_net / m
```

### Uniform gravity

`F_g = m g`, independent of position, velocity, and time. The default is
Earth-like `g = (0, -9.81) m/s^2` with the +y axis pointing upward.

### Constant applied force

`F = (Fx, Fy)`, constant in newtons, independent of the particle's state.
Combined with gravity the net force is constant, so the exact trajectory is
the constant-acceleration solution.

### Linear drag

`F_d = -b v`, with `b >= 0` in N per (m/s). The force opposes the
instantaneous velocity and is exactly zero at rest — no direction is invented.
Assumptions: low-Reynolds-number viscous regime, a medium at rest (wind is
deferred), constant coefficient.

For a constant driving force `F` along one axis, `m dv/dt = F - b v` has the
exact solution `v(t) = v_terminal + (v0 - v_terminal) exp(-gamma t)` with
`gamma = b/m` and `v_terminal = F/b` (signed; for gravity `F = -m g` so
`v_terminal = -m g / b`). The engine's explicit-Euler result is validated
against this reference in the test suite and in `examples/linear_drag.py`.

### Quadratic drag

`F_d = -(1/2) rho Cd A |v| v`, with fluid density `rho` (kg/m^3),
dimensionless drag coefficient `Cd`, and reference area `A` (m^2). The speed
multiplies the velocity vector directly, so zero speed yields exactly zero
force with no division and no NaNs. Assumptions: continuum fluid, constant
`rho`/`Cd`/`A`, drag depending only on the instantaneous velocity relative to
a medium at rest, no lift.

The terminal *speed* for vertical fall solves `m g = k v_t^2` with
`k = rho Cd A / 2`, giving `v_t = sqrt(2 m g / (rho Cd A))`. Terminal *speed*
is the magnitude; the terminal *velocity* points opposite to gravity.
`QuadraticDrag.terminal_speed(mass, gravity)` returns the magnitude. The fall
from rest follows the exact tanh solution, used as the validation reference
in `examples/quadratic_drag.py`.

### Friction and contact

Contact mechanics and the friction law are separate concepts:

- **Contact** (`PlaneSurface`, `horizontal_surface`, `inclined_surface`):
  a fixed rigid plane with an explicit unit normal and tangent. The normal
  force is derived from the perpendicular force balance `N = -(F_applied . n)`
  — never hard-coded to `m g` — so an incline automatically gives
  `N = m g cos(theta)` plus any applied perpendicular load.
- **Friction law** (`CoulombFriction`): kinetic `f_k = mu_k N` opposing the
  instantaneous sliding direction; static friction supplies exactly the
  tangential force required to keep the particle at rest, capped at
  `f_s,max = mu_s N`. A tangential force exactly at the threshold does not
  start motion (inclusive `<=` convention).

Assumptions: ideal rigid contact, fixed plane, constant coefficients with
`mu_k <= mu_s`, no rolling resistance, deformation, or adhesion. The particle
must stay on the surface: free flight, landing, and impacts raise
`ValueError` rather than being faked.

## Numerical method

Explicit Euler with a fixed user-selected timestep:

```text
v[n+1] = v[n] + a[n] dt
x[n+1] = x[n] + v[n] dt
```

The position update intentionally uses the old velocity; that is the defining
property of explicit Euler. No other integrator is implemented (V0.5 scope).

Known, documented consequences:

- For constant acceleration the velocity update is exact and the position
  carries a first-order `O(dt)` error.
- For linear drag, small `dt` relative to `m/b` is required; the tests report
  error reduction across timesteps rather than hiding stiffness.
- When friction stops a sliding particle inside one step, the exact piecewise
  velocity outcome is applied while the recorded position keeps the plain
  Euler update, overestimating the stopping distance by `O(dt)`. This is a
  convergent, tested approximation — never a hidden correction, and never a
  velocity clamp behind the user's back.

## Validation philosophy

Tests compare engine output against *independent* references, not the same
equation twice:

- exact analytical solutions (constant acceleration, linear-drag exponential
  solution, quadratic-drag tanh fall and its log-cosh distance),
- equilibrium properties (drag force at terminal speed balances weight),
- limiting cases (every drag parameter to zero, `b -> 0`, `mu -> 0`,
  zero-force laws),
- direction/symmetry properties (drag opposes velocity, reversing velocity
  reverses the force, rotation invariance of magnitudes),
- zero-velocity edge cases (no NaN/Inf, no invented directions),
- convergence (error strictly decreases as `dt` decreases; timestep ratios
  bounded near the expected first-order behavior),
- deterministic re-runs (identical inputs give bit-identical trajectories).

Approximate references are always labeled as such; nothing approximate is
presented as exact.

## Installation

Requires Python 3.10 or newer.

```bash
python -m pip install -e ".[test]"
```

This installs NumPy and Matplotlib, plus pytest for the test suite.

## Running tests

```bash
python -m pytest
```

The 0.2.0 suite contains 165 tests and passes. Ruff and strict mypy are also
clean (`ruff check .`, `mypy src/quasarlab`).

## Examples

```bash
python examples/projectile.py        # V0.1 gravity projectile (CSV + plot)
python examples/applied_force.py     # gravity + constant applied force
python examples/linear_drag.py       # fall with F = -b v vs exact solution
python examples/quadratic_drag.py    # terminal speed vs exact tanh solution
python examples/friction.py          # static threshold / kinetic slide
```

Each example prints engine-computed quantities (net force, terminal speed,
error against the analytical reference) and writes a plot under
`examples/output/`. All examples consume only the public engine API.

## Graphical interface

```bash
python quasarlab_gui.py
```

The Physics Lab window offers five scenarios from a dropdown — Projectile
(gravity), Applied force, Linear drag (falling), Quadratic drag (falling),
and Friction on a surface — each with its own labeled parameters and units.

- selecting a scenario rebuilds the parameter panel with sensible defaults,
- the plot updates automatically (debounced) as you edit any value,
- **Run simulation** forces an immediate recomputation,
- **Animate motion** replays the recorded trajectory,
- the info panel reports final position/velocity and the engine's force
  breakdown (individual forces, net force, acceleration) at the final state.

The GUI only collects parameters and displays engine results; every physics
number — including error and terminal-speed summaries — comes from the
engine or its analytical-reference functions.

## Building a Windows executable

Locally:

```bash
python -m pip install -e ".[test,dev,exe]"
python -m PyInstaller --onefile --windowed --name QuasarLab quasarlab_gui.py
dist\QuasarLab.exe --selftest
```

On GitHub, the `Build Windows EXE` workflow
(`.github/workflows/windows-exe.yml`) builds and smoke-tests a Windows
executable on every push and pull request and uploads it as a run artifact.

The `Build desktop executables` workflow
(`.github/workflows/build-binaries.yml`) additionally builds and smoke-tests
one-file executables for **Windows (x64)**, **Linux (x64, under Xvfb)**, and
**macOS (Apple Silicon)** on every push and pull request; each upload is a
separate artifact, and pushing a tag `v...` attaches all three to a GitHub
release. Linux needs the Tk runtime at build time; the workflow installs
`python3-tk` and runs the smoke test under `xvfb-run` because there is no
display server on CI runners.

## Architecture

```text
QuasarLab/
├── quasarlab_gui.py
├── src/quasarlab/
│   ├── physics/
│   │   ├── particle.py      # the 2D massive particle
│   │   ├── forces.py        # ForceLaw protocol; gravity, applied, drag laws
│   │   ├── contact.py       # plane contact + Coulomb friction (separate)
│   │   ├── systems.py       # ParticleSystem: composition, net force, a = F/m
│   │   └── analytical.py    # exact reference solutions
│   ├── numerical/
│   │   ├── state.py         # immutable (time, position, velocity) state
│   │   └── integrators.py   # explicit Euler
│   ├── simulation/
│   │   └── world.py         # fixed-dt loop, contact handling, Trajectory
│   └── visualization/
│       └── plotting.py      # renders recorded data; no physics
├── tests/
└── examples/
```

Layer responsibilities:

1. **Physics** defines particles, force laws, contact/friction, systems, and
   analytical laws. Force laws are pure: `force(particle, state)` returns a
   force vector, never mutating state, integrating, or advancing time.
2. **Numerical** defines the state and integration algorithms.
3. **Simulation** coordinates physics and numerics, optionally consults a
   contact model, and records trajectories.
4. **Visualization** plots recorded data and performs no physics.

The GUI consumes the public API and never contains physics equations.

## Current limitations

- One particle; multi-particle interaction is the V0.3 milestone.
- Only explicit Euler is implemented (further integrators belong to V0.5).
- Drag assumes a medium at rest; wind/relative velocity is deferred (the
  laws are documented so it can be added without redesign).
- The contact model supports fixed planes only: the particle must remain on
  the surface, and free flight/landing/impacts raise `ValueError`.
- Trajectories are recorded in memory; there is no persistence layer.
- The world runs in whole fixed timesteps; `duration` must be a multiple of
  `dt`.
- The GUI is an application script on top of the engine; it is exercised by
  its `--selftest` path rather than the unit-test suite.

## Future direction

The planned progression is:

1. V0.3: multiple interacting particles.
2. V0.4: N-body gravitational simulation.
3. V0.5: additional integrators and systematic comparison.
4. V0.6: energy, momentum, error, and convergence analysis tools.
5. V0.7: oscillators, orbital mechanics, collisions, and other systems.

A future AI-facing layer may translate natural-language requests into
structured simulation parameters, but it will always call this deterministic
engine rather than produce physics results itself.
