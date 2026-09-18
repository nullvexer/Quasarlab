"""The simulation loop: coordinate physics and numerics, record results."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quasarlab._validation import as_nonnegative_float, as_positive_float
from quasarlab.numerical.integrators import euler_step
from quasarlab.numerical.state import State
from quasarlab.physics.contact import ContactModel
from quasarlab.physics.systems import ForceEvaluation, ParticleSystem


@dataclass(frozen=True)
class Trajectory:
    """A recorded trajectory.

    ``time`` has shape (N,); ``position`` and ``velocity`` have shape (N, 2).
    """

    time: NDArray[np.float64]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        position = np.asarray(self.position, dtype=float)
        velocity = np.asarray(self.velocity, dtype=float)
        if time.ndim != 1:
            raise ValueError(f"time must be 1-D, got shape {time.shape}")
        if len(time) == 0:
            raise ValueError("trajectory must contain at least one sample")
        if position.shape != (len(time), 2):
            raise ValueError(f"position must have shape {(len(time), 2)}, got {position.shape}")
        if velocity.shape != (len(time), 2):
            raise ValueError(f"velocity must have shape {(len(time), 2)}, got {velocity.shape}")
        if not (
            np.all(np.isfinite(time))
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(velocity))
        ):
            raise ValueError("trajectory arrays must contain only finite values")
        if np.any(time < 0.0):
            raise ValueError("trajectory times must be nonnegative")
        object.__setattr__(self, "time", time.copy())
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(self, "velocity", velocity.copy())

    @property
    def x(self) -> NDArray[np.float64]:
        """Return the x coordinate of every sample."""
        return self.position[:, 0]

    @property
    def y(self) -> NDArray[np.float64]:
        """Return the y coordinate of every sample."""
        return self.position[:, 1]

    @property
    def vx(self) -> NDArray[np.float64]:
        """Return the x velocity of every sample."""
        return self.velocity[:, 0]

    @property
    def vy(self) -> NDArray[np.float64]:
        """Return the y velocity of every sample."""
        return self.velocity[:, 1]

    def __len__(self) -> int:
        return len(self.time)


@dataclass
class World:
    """A deterministic simulation of one particle.

    ``run`` always starts from the particle in ``system``.  ``duration`` must
    be a nonnegative multiple of ``dt``; ``run_steps`` can be used for an exact
    number of whole timesteps.

    ``contact`` optionally attaches a contact model (e.g. a frictional plane).
    With contact, each step evaluates the net applied force, asks the contact
    for its reaction (normal + friction), integrates, and then applies the
    contact's discrete-time kinematic constraint.  Without contact the loop is
    exactly the V0.1 loop.

    V0.1 scopes this to exactly one particle; multi-particle systems are
    deferred to the V0.3 roadmap item.
    """

    system: ParticleSystem
    dt: float
    integrator: Callable[[State, NDArray[np.float64], float], State] = euler_step
    contact: ContactModel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.system, ParticleSystem):
            raise TypeError(f"system must be a ParticleSystem, got {type(self.system).__name__}")
        self.dt = as_positive_float(self.dt, "dt")
        if not callable(self.integrator):
            raise TypeError("integrator must be callable")
        if self.contact is not None and not isinstance(self.contact, ContactModel):
            raise TypeError("contact must implement the ContactModel protocol")

    @property
    def initial_state(self) -> State:
        """Return the initial state derived from the system's particle."""
        particle = self.system.particle
        return State(0.0, particle.position, particle.velocity)

    def evaluate_forces(self, state: State) -> ForceEvaluation:
        """Evaluate instantaneous loads, including contact, without advancing time."""
        contributions = self.system.force_contributions(state)
        total = np.zeros(2, dtype=float)
        for _, force in contributions:
            total += force
        if self.contact is not None:
            contact_forces = self.contact.force_contributions(
                self.system.particle, state, total
            )
            contributions += contact_forces
            for _, force in contact_forces:
                total += force
        return ForceEvaluation(contributions, total, total / self.system.particle.mass)

    def run_steps(self, n_steps: int) -> Trajectory:
        """Run exactly ``n_steps`` whole timesteps and return the trajectory."""
        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise ValueError(f"n_steps must be an integer, got {n_steps!r}")
        if n_steps < 0:
            raise ValueError(f"n_steps must be nonnegative, got {n_steps}")

        state = self.initial_state
        if self.contact is not None:
            self.contact.reaction(self.system.particle, state, self.system.net_force(state))
        times = [state.time]
        positions = [state.position.copy()]
        velocities = [state.velocity.copy()]
        for _ in range(n_steps):
            if self.contact is None:
                acceleration = self.system.acceleration(state)
                state = self.integrator(state, acceleration, self.dt)
            else:
                particle = self.system.particle
                applied = self.system.net_force(state)
                contact_force = self.contact.reaction(particle, state, applied)
                acceleration = (applied + contact_force) / particle.mass
                previous_state = state
                state = self.integrator(state, acceleration, self.dt)
                state = self.contact.constrain(particle, previous_state, state, applied, self.dt)
            if not isinstance(state, State):
                raise TypeError(f"integrator must return a State, got {type(state).__name__}")
            times.append(state.time)
            positions.append(state.position.copy())
            velocities.append(state.velocity.copy())
        return Trajectory(np.asarray(times), np.asarray(positions), np.asarray(velocities))

    def run(self, duration: float) -> Trajectory:
        """Run for ``duration`` seconds using whole timesteps of size ``dt``.

        ``duration`` must be a nonnegative multiple of ``dt`` within normal
        floating-point tolerance.  Invalid input raises ``ValueError``.
        """
        duration = as_nonnegative_float(duration, "duration")
        n_steps = round(duration / self.dt)
        tolerance = 1e-9 * max(1.0, duration)
        if not math.isclose(n_steps * self.dt, duration, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"duration {duration!r} is not a whole-number multiple of dt {self.dt!r}"
            )
        return self.run_steps(n_steps)
