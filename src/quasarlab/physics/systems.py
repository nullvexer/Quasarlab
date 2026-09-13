"""Physical systems: a particle together with the force laws acting on it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quasarlab._validation import as_vector
from quasarlab.numerical.state import State
from quasarlab.physics.forces import ForceLaw
from quasarlab.physics.particle import Particle


@dataclass
class ParticleSystem:
    """A particle and all force laws acting on it.

    Newton's second law is applied here:

        F_total = sum(force laws)
        a       = F_total / m

    V0.1 scopes this to exactly one particle; multi-particle systems are
    deferred to the V0.3 roadmap item.
    """

    particle: Particle
    forces: tuple[ForceLaw, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.particle, Particle):
            raise TypeError(f"particle must be a Particle, got {type(self.particle).__name__}")
        self.forces = tuple(self.forces)
        for index, law in enumerate(self.forces):
            if not callable(getattr(law, "force", None)):
                raise TypeError(
                    f"force law {index} must provide a callable force(particle, state)"
                )

    def net_force(self, state: State) -> NDArray[np.float64]:
        """Return the total force on the particle at ``state`` in newtons."""
        total: NDArray[Any] = np.zeros(2, dtype=float)
        for law in self.forces:
            force = as_vector(law.force(self.particle, state), "force")
            total = np.asarray(total + force, dtype=float)
        return total

    def acceleration(self, state: State) -> NDArray[np.float64]:
        """Return the acceleration ``a = F_total / m`` in m/s^2."""
        return self.net_force(state) / self.particle.mass
