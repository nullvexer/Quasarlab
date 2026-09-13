"""The numerical state advanced by integrators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quasarlab._validation import as_nonnegative_float, as_vector


@dataclass(frozen=True)
class State:
    """A 2D kinematic state at one instant.

    Units:
        time -> seconds
        position -> meters, shape (2,)
        velocity -> meters/second, shape (2,)
    """

    time: float
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", as_nonnegative_float(self.time, "time"))
        object.__setattr__(self, "position", as_vector(self.position, "position"))
        object.__setattr__(self, "velocity", as_vector(self.velocity, "velocity"))
