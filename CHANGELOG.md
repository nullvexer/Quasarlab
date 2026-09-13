# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

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
