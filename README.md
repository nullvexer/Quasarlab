# QuasarLab

QuasarLab is a deterministic computational physics laboratory. Its physics
engine is the source of truth: results are produced by explicit computation,
not by heuristics or generated guesses.

This repository currently contains **Version 0.1.1**, a deliberately small
foundation: one 2D particle moving under uniform gravitational acceleration,
integrated with explicit Euler and validated against the exact analytical
solution.

## Current capabilities

- 2D particle with mass, position, and velocity in SI units
- Uniform gravitational force `F_g = m g`
- Net force and acceleration through Newton's second law `a = F / m`
- Explicit Euler numerical integration
- Fixed-timestep simulation loop with trajectory recording
- Exact constant-acceleration analytical reference solution
- Numerical-vs-analytical error checks and convergence tests
- Matplotlib trajectory visualization
- Tkinter graphical interface (`quasarlab_gui.py`) on top of the engine
- GitHub Actions workflow that packages a Windows `.exe`
- Clear rejection of invalid physical and numerical input

## Physics implemented

The V0.1 model is Newton's second law with uniform gravity:

```text
F = m a
F_g = m g
a = F_g / m = g
```

The default gravity is Earth-like:

```text
g = (0, -9.81) m/s^2
```

with the +y axis pointing upward. SI units are used throughout:

```text
position      -> meters
velocity      -> meters/second
acceleration  -> meters/second^2
force         -> newtons
mass          -> kilograms
time          -> seconds
```

## Numerical method

Version 0.1 uses explicit Euler integration:

```text
v[n+1] = v[n] + a[n] dt
x[n+1] = x[n] + v[n] dt
```

The position update intentionally uses the old velocity, which is the defining
property of explicit Euler. No other integrator is implemented yet.

For constant acceleration, the Euler velocity update is exact, while the
position carries a first-order error proportional to `dt`.

## Analytical validation

For constant acceleration, the exact reference solution is:

```text
x(t) = x0 + v0 t + (1/2) a t^2
v(t) = v0 + a t
```

The automated tests compare numerical trajectories against this solution for
`timestep = 0.1`, `0.01`, and `0.001` seconds and require the numerical error
to decrease as the timestep decreases.

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

The 0.1.1 suite contains 77 tests and passes.

## Running the projectile example

```bash
python examples/projectile.py
```

Optional flags:

```bash
python examples/projectile.py --speed 20 --angle 45 --dt 0.01
python examples/projectile.py --show
```

The example writes:

```text
examples/output/projectile_trajectory.csv
examples/output/projectile.png
```

The CSV contains `time`, `x`, `y`, `vx`, and `vy`. The plot shows the numerical
trajectory together with the analytical trajectory. With the default settings,
the example reports a maximum position error of about `0.142 m` for
`dt = 0.01 s`.

## Graphical interface

```bash
python quasarlab_gui.py
```

A small Tkinter window opens:

- set launch speed, angle, mass, gravity magnitude, and timestep,
- the plot updates automatically (debounced) as you edit any value,
- press **Run simulation** to force an immediate recomputation,
- press **Animate flight** to replay the recorded trajectory,
- the plot overlays the analytical solution and reports the numerical error.

While typing, invalid or extreme input is reported in the status label instead
of a dialog, and the last valid plot stays on screen.

The interface only collects parameters and displays results; every physics
number comes from the QuasarLab engine.

## Building a Windows executable

Locally:

```bash
python -m pip install -e ".[test,dev,exe]"
python -m PyInstaller --onefile --windowed --name QuasarLab quasarlab_gui.py
dist\QuasarLab.exe --selftest
```

On GitHub, the `Build Windows EXE` workflow
(`.github/workflows/windows-exe.yml`) builds and smoke-tests the same
executable on every push and pull request and uploads it as a run artifact.
Pushing a tag `v...` also attaches the executable to a GitHub release.
Download it from the repository's **Actions** (or **Releases**) page.

## Architecture

```text
QuasarLab/
├── README.md
├── pyproject.toml
├── .gitignore
├── quasarlab_gui.py
├── src/quasarlab/
│   ├── physics/
│   │   ├── particle.py
│   │   ├── forces.py
│   │   ├── systems.py
│   │   └── analytical.py
│   ├── numerical/
│   │   ├── state.py
│   │   └── integrators.py
│   ├── simulation/
│   │   └── world.py
│   └── visualization/
│       └── plotting.py
├── tests/
└── examples/
```

Layer responsibilities:

1. **Physics** defines particles, force laws, systems, and analytical laws.
2. **Numerical** defines the state and integration algorithms.
3. **Simulation** coordinates physics and numerics and records trajectories.
4. **Visualization** plots recorded data and performs no physics calculations.

## Current limitations

- Only uniform gravity is implemented.
- Only explicit Euler is implemented.
- The simulation supports one particle.
- Trajectories are recorded in memory; there is no persistence layer.
- The world runs in whole fixed timesteps; `duration` must be a multiple of
  `dt`.
- There is no collision, ground, drag, rotation, or energy analysis yet.
- The GUI is an application script on top of the engine; it is not covered by
  the unit-test suite.

## Future direction

The planned progression is:

1. V0.2: general force framework, including drag, springs, and user forces.
2. V0.3: multiple interacting particles.
3. V0.4: N-body gravitational simulation.
4. V0.5: additional integrators and systematic comparison.
5. V0.6: energy, momentum, error, and convergence analysis tools.
6. V0.7: oscillators, orbital mechanics, collisions, and other systems.

A future AI-facing layer may translate natural-language requests into
structured simulation parameters, but it will always call this deterministic
engine rather than produce physics results itself.
