# Changelog

All notable changes to this project are documented in this file.

## [0.2.1] — Engine hardening: immutability and contact abstraction

### Added

- `CoulombFriction`, a public friction law separated from contact mechanics:
  it consumes a supplied normal load and returns the signed tangential Coulomb
  force (kinetic branch when sliding, exact static branch at rest), so future
  contact models can reuse the law without inheriting plane geometry.

### Changed

- Every built-in force law (`UniformGravity`, `ConstantForce`, `LinearDrag`,
  `QuadraticDrag`) and `PlaneSurface` is now a frozen dataclass whose vector
  fields are write-protected; laws and surfaces cannot be mutated into
  invalid physics after construction.
- `as_vector` returns write-protected arrays, and validation now rejects
  booleans (scalar and vector components) and blank labels everywhere, so
  `True` can no longer masquerade as a measurement.
- `World` depends only on the generic `ContactModel` protocol: the protocol
  gained `force_contributions`, and `evaluate_forces` no longer special-cases
  `PlaneSurface`. Custom contact models now report labelled contributions on
  equal footing with the built-in plane.

### Validation

- New contract tests (165 → 184 total) assert frozen dataclass status,
  write-protected law/surface/state arrays, boolean and blank-label
  rejection, and that `World.evaluate_forces` stays correct through the
  generic protocol.
- `quadratic_drag_fall_distance` now evaluates `ln(cosh x)` through two exact
  identities chosen by argument size (`tanh`-based for short times,
  `logaddexp`-based for long times); regression tests pin the ballistic
  short-time limit `d ~ g t^2 / 2` with zero absolute tolerance and the
  finite long-time limit, for scalar and mixed-array input.

### Fixed

- `quadratic_drag_fall_distance` lost the short-time ballistic limit to
  floating-point cancellation (it returned exactly `0` for `t = 1e-10` where
  the exact distance is `5e-20`); the size-dependent evaluation restores full
  relative accuracy on both ends of the time range.

### Limitations

- Unchanged from 0.2.0: surface contact only (no free flight/impacts),
  explicit Euler only, drag assumes a medium at rest.

## [0.2.0] — Force framework and realistic forces

### Added

- Force-law composition: `ParticleSystem` accepts any number of force laws and
  evaluates `F_net = sum(F_i)`, `a = F_net / m`.
- `ConstantForce`: a constant applied force `(Fx, Fy)` in newtons.
- `LinearDrag`: `F_d = -b v` with `b >= 0`; zero force at rest.
- `QuadraticDrag`: `F_d = -(1/2) rho Cd A |v| v`; zero force at rest with no
  division by speed, plus `terminal_speed(mass, gravity)` returning the
  magnitude `sqrt(m g / k)` (documented as a speed, not a signed velocity).
- Contact mechanics separate from the friction law: `PlaneSurface` (fixed
  plane, explicit normal/tangent) consumes a `CoulombFriction` law and derives
  the normal force from the perpendicular force balance (`N = -(F . n)`),
  never hard-coded to `m g`. `horizontal_surface` and `inclined_surface`
  construct the common geometries.
- Static friction supplies exactly the tangential force required to hold the
  particle (capped at `mu_s N`); kinetic friction `mu_k N` opposes sliding.
  Kinetic friction can stop a sliding particle but never reverse it (see
  Limitations for the documented discrete-time handling).
- `World(contact=...)`: optional contact model integration in the simulation
  loop; without contact the loop is byte-for-byte the V0.1 loop.
- `World.evaluate_forces(state)`: engine-side instantaneous force breakdown
  (`contributions`, `net_force`, `acceleration`) including contact reactions;
  the basis for free-body transparency in the GUI.
- Force-law `label` fields and `ParticleSystem.force_contributions` for
  per-law reporting; user laws without labels fall back to the class name.
- Exact analytical references: 1D linear-drag velocity/position (constant
  driving force), and quadratic-drag fall speed (tanh) and distance
  (log-cosh, evaluated overflow-free via `logaddexp`).
- GUI scenarios: projectile, applied force, linear drag, quadratic drag, and
  friction on a surface. The GUI collects parameters and displays engine
  results only; every reported number comes from the engine or its
  analytical-reference functions.
- Examples: `applied_force.py`, `linear_drag.py`, `quadratic_drag.py`,
  `friction.py` alongside the existing `projectile.py`.

### Changed

- `plot_trajectory` and the new `plot_component_vs_time` accept an existing
  Matplotlib `Axes` for GUI embedding; behavior without `ax` is unchanged.
- `quasarlab_gui.py` is structured as scenario objects (`build`, `plot`,
  `report`) over one generic `PhysicsLab` window instead of a projectile-only
  script; the previous live-debounced updates, animation, and `--selftest`
  behavior are preserved.
- Public API additions exported from `quasarlab` (`ConstantForce`,
  `LinearDrag`, `QuadraticDrag`, `ForceLaw`, contact constructors, drag
  analytical references). The V0.1 imports remain valid.

### Validation

- Regression suite retained (77 tests in 0.1.1); suite now totals 165 tests.
- Linear drag: numerical velocity/position vs the exact exponential solution
  (fall, and 2D with gravity + applied force), terminal-velocity limit
  `v_t = -F/b`, monotone approach to terminal speed from above, and
  first-order error reduction across timesteps.
- Quadratic drag: fall speed vs the exact tanh solution and distance vs the
  exact log-cosh solution; equilibrium check `k v_t^2 = m g` at terminal
  speed; monotone approach to terminal speed; error reduction across
  timesteps; finite long-time distance reference.
- Constant force: exact constant-acceleration validation with gravity and an
  applied force; direction-reversal and rotation-invariance property tests.
- Limiting cases: each drag parameter individually drives the force to zero;
  zero drag reproduces the ballistic trajectory exactly; adding a zero-force
  law does not change any recorded sample.
- Zero-velocity edge cases for every velocity-dependent law: no NaN/Inf and
  no invented directions at `|v| = 0`.
- Friction: static cases A–F (rest, balance below threshold, inclusive
  threshold convention, sliding onset, kinetic deceleration to rest without
  jitter, exact piecewise restart), incline normal force `m g cos(theta)` and
  exact sliding acceleration, convergence of stopping distance across
  timesteps, rejection of normal velocity and of particles off the surface.
- Deterministic re-run equality is asserted for gravity and friction runs.

### Fixed

- `quadratic_drag_fall_distance` overflowed (`inf`) for long times even
  though the exact distance is finite; it now uses `logaddexp` and returns
  the finite exact value.
- The friction run did not validate the initial state's contact conditions
  when `n_steps = 0`; contact validity is now checked before recording.
- Normal-velocity (impact) states silently passed friction evaluation; they
  now raise a clear error stating impacts are unsupported in V0.2.

### Limitations

- The contact model requires the particle to stay on the surface: free
  flight, landing, and impacts raise `ValueError` by design.
- With explicit Euler, a stop inside a timestep uses the documented exact
  piecewise velocity with the plain Euler position update (stopping distance
  overestimated by `O(dt)`; convergent, tested).
- Drag assumes a medium at rest; wind (relative-velocity) support is deferred.
- Only explicit Euler is implemented (integrator work belongs to V0.5).

## [0.1.1] — Graphical interface and Windows executable

### Added

- Tkinter graphical interface (`quasarlab_gui.py`) that runs simulations
  through the deterministic engine and replays recorded trajectories.
- Live parameter updates in the GUI: the plot refreshes automatically
  (debounced) while inputs change; invalid input is reported in the status
  label instead of a dialog.
- `exe` optional dependency group (PyInstaller).
- GitHub Actions workflow `windows-exe.yml` that builds and smoke-tests a
  one-file Windows executable, uploads it as an artifact, and attaches it to
  releases on version tags.

### Changed

- Type annotations use `numpy.typing.NDArray[np.float64]` so strict mypy does
  not depend on the installed NumPy stub generation.
- CI runs `mypy` on the Python 3.12 matrix leg (NumPy stubs older than 2.5
  are not clean under mypy 2.x); ruff and pytest still run on every leg.

## [0.1.0] — Initial release

- 2D particle with mass, position, and velocity (SI units).
- Uniform gravitational force and Newton's-second-law acceleration.
- Explicit Euler integrator.
- Fixed-timestep simulation loop (`World`) with trajectory recording.
- Exact analytical reference solution for constant acceleration.
- Numerical-vs-analytical error and convergence tests.
- Matplotlib trajectory visualization.
- Strict validation of physical and numerical inputs.
