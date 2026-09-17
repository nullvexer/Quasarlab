"""Physical systems: a particle together with the force laws acting on it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quasarlab._validation import as_vector
from quasarlab.numerical.state import State
from quasarlab.physics.forces import ForceLaw
from quasarlab.physics.particle import Particle


@dataclass(frozen=True)
class ForceEvaluation:
    """Instantaneous force contributions, their sum (N), and acceleration (m/s²)."""

    contributions: tuple[tuple[str, NDArray[np.float64]], ...]
    net_force: NDArray[np.float64]
    acceleration: NDArray[np.float64]


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
        total: NDArray[np.float64] = np.zeros(2, dtype=float)
        for _, force in self.force_contributions(state):
            total = np.asarray(total + force, dtype=float)
        return total

    def force_contributions(
        self, state: State
    ) -> tuple[tuple[str, NDArray[np.float64]], ...]:
        """Return ``(label, force)`` pairs for every law, in composition order.

        The labels come from each law's ``label`` attribute (falling back to
        the class name for user-defined laws).  The forces sum exactly to
        :meth:`net_force`; this is the engine-side force breakdown used for
        free-body transparency.
        """
        contributions = []
        for law in self.forces:
            label = str(getattr(law, "label", type(law).__name__))
            contributions.append((label, as_vector(law.force(self.particle, state), "force")))
        return tuple(contributions)

    def acceleration(self, state: State) -> NDArray[np.float64]:
        """Return the acceleration ``a = F_total / m`` in m/s^2."""
        return self.net_force(state) / self.particle.mass
