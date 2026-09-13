# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Tkinter graphical interface (`quasarlab_gui.py`) that runs simulations
  through the deterministic engine and replays recorded trajectories.
- `exe` optional dependency group (PyInstaller).
- GitHub Actions workflow `windows-exe.yml` that builds and smoke-tests a
  one-file Windows executable, uploads it as an artifact, and attaches it to
  releases on version tags.

## [0.1.0] — Initial release

- 2D particle with mass, position, and velocity (SI units).
- Uniform gravitational force and Newton's-second-law acceleration.
- Explicit Euler integrator.
- Fixed-timestep simulation loop (`World`) with trajectory recording.
- Exact analytical reference solution for constant acceleration.
- Numerical-vs-analytical error and convergence tests.
- Matplotlib trajectory visualization.
- Strict validation of physical and numerical inputs.
