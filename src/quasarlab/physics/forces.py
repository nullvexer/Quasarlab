"""Force laws.

V0.1 implements uniform gravity, for which:

    F_g = m g
    a   = F_g / m = g

A force law is any object exposing ``force(particle, state)`` and returning a
2-component force vector in newtons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from quasarlab._validation import as_positive_float, as_vector
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


@dataclass
class UniformGravity:
    """A uniform gravitational field.

    ``g`` is the acceleration vector in m/s^2.  The default is Earth-like
    gravity ``(0, -9.81)`` with the +y axis pointing upward.
    """

    g: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, -9.81]))

    def __post_init__(self) -> None:
        self.g = as_vector(self.g, "g")

    def force(self, particle: Particle, state: State) -> NDArray[np.float64]:
        """Return the force on ``particle`` in newtons.

        ``state`` is unused by uniform gravity but is part of the force-law
        interface so later laws can depend on position or velocity.
        """
        return gravitational_force(particle.mass, self.g)

    def acceleration(self, particle: Particle, state: State) -> NDArray[np.float64]:
        """Return the acceleration ``a = g`` in m/s^2."""
        return self.g.copy()
