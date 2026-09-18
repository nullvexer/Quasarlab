"""Force laws.

Every force law is a pure description of a physical law: it exposes
``force(particle, state)`` and returns a 2-component force vector in newtons.
Force laws never mutate state, never integrate, and never advance time; the
numerical layer owns integration and the systems layer owns composition

    F_net = sum(force laws)
    a     = F_net / m

All laws carry a ``label`` so the force breakdown can report individual
contributions (free-body transparency).

Every built-in law is a frozen dataclass whose vector fields are
write-protected: a force law is an immutable description of a physical law
and cannot be mutated into invalid physics after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from quasarlab._validation import (
    as_label,
    as_nonnegative_float,
    as_positive_float,
    as_vector,
)
from quasarlab.numerical.state import State
from quasarlab.physics.particle import Particle


@runtime_checkable
class ForceLaw(Protocol):
    """The interface every force law must satisfy.

    A force law is any object exposing ``force(particle, state)`` and
    returning a 2-component force vector in newtons. This formalizes, as a
    type, the informal contract already documented above.
    """

    def force(self, particle: Particle, state: State) -> NDArray[np.float64]: ...


def gravitational_force(mass: float, gravity: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the gravitational force ``m g`` in newtons."""
    mass = as_positive_float(mass, "mass")
    gravity = as_vector(gravity, "gravity")
    return mass * gravity


@dataclass(frozen=True)
class UniformGravity:
    """A uniform gravitational field.

    ``g`` is the acceleration vector in m/s^2.  The default is Earth-like
    gravity ``(0, -9.81)`` with the +y axis pointing upward.

    F = m g, independent of position, velocity, and time.
    """

    g: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, -9.81]))
    label: str = "uniform gravity"

    def __post_init__(self) -> None:
        object.__setattr__(self, "g", as_vector(self.g, "g"))
        object.__setattr__(self, "label", as_label(self.label, "label"))

    def force(self, particle: Particle, state: State) -> NDArray[np.float64]:
        """Return the force on ``particle`` in newtons.

        ``state`` is unused by uniform gravity but is part of the force-law
        interface so other laws can depend on position or velocity.
        """
        return gravitational_force(particle.mass, self.g)

    def acceleration(self, particle: Particle, state: State) -> NDArray[np.float64]:
        """Return the acceleration ``a = g`` in m/s^2."""
        return self.g.copy()


@dataclass(frozen=True)
class ConstantForce:
    """A force of constant magnitude and direction.

    F = (F_x, F_y), constant in newtons and independent of the particle's
    mass, position, velocity, and time.  Useful for applied/pushed loads;
    combine with other laws (gravity, drag, friction) through the system.
    """

    force_value: NDArray[np.float64]
    label: str = "applied force"

    def __post_init__(self) -> None:
        object.__setattr__(self, "force_value", as_vector(self.force_value, "force"))
        object.__setattr__(self, "label", as_label(self.label, "label"))

    def force(self, particle: Particle, state: State) -> NDArray[np.float64]:
        """Return the constant force vector in newtons."""
        return self.force_value.copy()


@dataclass(frozen=True)
class LinearDrag:
    """Linear (viscous) drag: F_d = -b v, with b >= 0 in N per (m/s).

    The force always opposes the instantaneous velocity; at rest it is exactly
    zero (no direction is invented).

    Model assumptions: low-Reynolds-number (viscous-dominated) regime, a
    medium at rest, and a drag coefficient independent of speed.  The b = 0
    limit is a valid zero-force law.
    """

    b: float
    label: str = "linear drag"

    def __post_init__(self) -> None:
        object.__setattr__(self, "b", as_nonnegative_float(self.b, "b"))
        object.__setattr__(self, "label", as_label(self.label, "label"))

    def force(self, particle: Particle, state: State) -> NDArray[np.float64]:
        """Return ``-b v`` in newtons; zero vector at zero velocity."""
        return -self.b * state.velocity


@dataclass(frozen=True)
class QuadraticDrag:
    """Quadratic drag: F_d = -(1/2) rho C_d A |v| v.

    ``density`` is the fluid density in kg/m^3, ``drag_coefficient`` is the
    dimensionless C_d, and ``area`` is the reference cross-sectional area in
    m^2.  The force magnitude is k |v|^2 with k = rho C_d A / 2, directed
    opposite to the velocity.  The speed multiplies the velocity directly, so
    zero velocity yields exactly zero force with no division and no NaNs.

    Model assumptions: continuum fluid, constant density/C_d/area, drag
    depending only on instantaneous velocity relative to a medium at rest
    (no wind yet), and no lift.
    """

    density: float
    drag_coefficient: float
    area: float
    label: str = "quadratic drag"

    def __post_init__(self) -> None:
        object.__setattr__(self, "density", as_nonnegative_float(self.density, "density"))
        object.__setattr__(
            self,
            "drag_coefficient",
            as_nonnegative_float(self.drag_coefficient, "drag_coefficient"),
        )
        object.__setattr__(self, "area", as_nonnegative_float(self.area, "area"))
        object.__setattr__(self, "label", as_label(self.label, "label"))

    @property
    def drag_factor(self) -> float:
        """Return k = (1/2) rho C_d A in kg/m."""
        return 0.5 * self.density * self.drag_coefficient * self.area

    def force(self, particle: Particle, state: State) -> NDArray[np.float64]:
        """Return ``-k |v| v`` in newtons; zero vector at zero velocity."""
        speed = float(np.linalg.norm(state.velocity))
        return -(self.drag_factor * speed) * state.velocity

    def terminal_speed(self, mass: float, gravity: float = 9.81) -> float:
        """Return the terminal speed for vertical fall in m/s (a magnitude).

        At terminal speed the drag magnitude balances the weight:

            m g = k v_t^2   =>   v_t = sqrt(m g / k)

        The terminal *velocity* points along gravity; this method
        returns the speed (magnitude), not a signed velocity.
        """
        mass = as_positive_float(mass, "mass")
        gravity = as_positive_float(gravity, "gravity")
        k = self.drag_factor
        if k <= 0.0:
            raise ValueError(
                "terminal speed requires positive drag: density, drag_coefficient"
                " and area must not all be zero"
            )
        return float(np.sqrt(mass * gravity / k))
