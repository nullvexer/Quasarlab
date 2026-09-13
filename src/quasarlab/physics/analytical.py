"""Exact analytical references for constant-acceleration motion.

For constant acceleration ``a``:

    x(t) = x0 + v0 t + (1/2) a t^2
    v(t) = v0 + a t

These are mathematical references used to validate numerical results.
"""

from __future__ import annotations

import numpy as np

from quasarlab._validation import as_vector


def _time_array(t: object) -> np.ndarray:
    """Return ``t`` as a scalar or 1-D finite float array."""
    try:
        array = np.asarray(t, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"t must be numeric, got {t!r}") from exc
    if array.ndim > 1:
        raise ValueError(f"t must be scalar or 1-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"t must contain only finite values, got {t!r}")
    return array


def position(
    t: object, initial_position: object, initial_velocity: object, acceleration: object
) -> np.ndarray:
    """Return the exact position under constant acceleration.

    ``t`` may be scalar or 1-D.  The return shape is (2,) for scalar input and
    (N, 2) for array input.
    """
    t_array = _time_array(t)
    x0 = as_vector(initial_position, "initial_position")
    v0 = as_vector(initial_velocity, "initial_velocity")
    a = as_vector(acceleration, "acceleration")
    if t_array.ndim == 0:
        result = x0 + v0 * t_array + 0.5 * a * t_array * t_array
    else:
        result = x0 + v0 * t_array[:, None] + 0.5 * a * (t_array[:, None] ** 2)
    return np.asarray(result, dtype=float)


def velocity(t: object, initial_velocity: object, acceleration: object) -> np.ndarray:
    """Return the exact velocity under constant acceleration.

    ``t`` may be scalar or 1-D.  The return shape is (2,) for scalar input and
    (N, 2) for array input.
    """
    t_array = _time_array(t)
    v0 = as_vector(initial_velocity, "initial_velocity")
    a = as_vector(acceleration, "acceleration")
    if t_array.ndim == 0:
        result = v0 + a * t_array
    else:
        result = v0 + a * t_array[:, None]
    return np.asarray(result, dtype=float)
