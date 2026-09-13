"""Shared validation helpers for QuasarLab.

This module keeps invalid scientific input out of the physics and numerical
layers.  Values are never silently repaired; invalid input raises clearly.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def as_float(value: object, name: str) -> float:
    """Return ``value`` as a finite float, or raise.

    ``value`` is deliberately typed as ``object``: this is a validation
    boundary meant to accept arbitrary caller input and either convert it
    or raise a clear error. The cast below tells the type checker what the
    surrounding try/except already enforces at runtime.
    """
    try:
        scalar = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar!r}")
    return scalar


def as_nonnegative_float(value: object, name: str) -> float:
    """Return ``value`` as a finite nonnegative float, or raise."""
    scalar = as_float(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {scalar!r}")
    return scalar


def as_positive_float(value: object, name: str) -> float:
    """Return ``value`` as a finite strictly positive float, or raise."""
    scalar = as_float(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {scalar!r}")
    return scalar


def as_vector(value: object, name: str) -> NDArray[np.float64]:
    """Return ``value`` as an owned finite float64 vector of shape (2,)."""
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 2-component numeric vector, got {value!r}") from exc
    if vector.shape != (2,):
        raise ValueError(f"{name} must be a 2-component vector, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"all components of {name} must be finite, got {vector!r}")
    return vector.copy()
