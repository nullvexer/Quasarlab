"""The physical object being simulated: a massive 2D point particle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasarlab._validation import as_positive_float, as_vector


@dataclass
class Particle:
    """A 2D point particle.

    Units:
        mass -> kilograms
        position -> meters, shape (2,)
        velocity -> meters/second, shape (2,)
    """

    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        self.mass = as_positive_float(self.mass, "mass")
        self.position = as_vector(self.position, "position")
        self.velocity = as_vector(self.velocity, "velocity")
