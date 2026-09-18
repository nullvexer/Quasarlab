"""Exact analytical references used to validate numerical results.

These are closed-form mathematical solutions, not approximations:

* constant acceleration ( ballistic motion / constant net force),
* 1D motion with linear drag and a constant driving force,
* 1D vertical fall from rest with quadratic drag.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from quasarlab._validation import as_float, as_positive_float, as_vector


def _time_array(t: object) -> NDArray[np.float64]:
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
) -> NDArray[np.float64]:
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


def velocity(t: object, initial_velocity: object, acceleration: object) -> NDArray[np.float64]:
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


def linear_drag_velocity(
    t: object, v0: float, gamma: float, v_terminal: float
) -> float | NDArray[np.float64]:
    """Return the exact 1D velocity with linear drag and a constant driving force.

    For ``m dv/dt = F - b v`` (quantities signed along one axis) the exact
    solution is ``v(t) = v_terminal + (v0 - v_terminal) exp(-gamma t)`` with
    ``gamma = b/m`` and ``v_terminal = F/b``.  ``gamma`` must be positive.
    """
    t_array = _time_array(t)
    v0 = as_float(v0, "v0")
    gamma = as_positive_float(gamma, "gamma")
    v_terminal = as_float(v_terminal, "v_terminal")
    if t_array.ndim == 0:
        return v_terminal + (v0 - v_terminal) * math.exp(-gamma * float(t_array))
    return v_terminal + (v0 - v_terminal) * np.exp(-gamma * t_array)


def linear_drag_position(
    t: object, x0: float, v0: float, gamma: float, v_terminal: float
) -> float | NDArray[np.float64]:
    """Return the exact 1D position with linear drag and a constant driving force.

    ``x(t) = x0 + v_terminal t + (v0 - v_terminal)(1 - exp(-gamma t)) / gamma``.
    """
    t_array = _time_array(t)
    x0 = as_float(x0, "x0")
    v0 = as_float(v0, "v0")
    gamma = as_positive_float(gamma, "gamma")
    v_terminal = as_float(v_terminal, "v_terminal")
    if t_array.ndim == 0:
        elapsed = float(t_array)
        return x0 + v_terminal * elapsed + (v0 - v_terminal) * (
            1.0 - math.exp(-gamma * elapsed)
        ) / gamma
    approach = (v0 - v_terminal) * (1.0 - np.exp(-gamma * t_array)) / gamma
    return x0 + v_terminal * t_array + approach


def quadratic_drag_fall_speed(
    t: object, v_terminal: float, g: float
) -> float | NDArray[np.float64]:
    """Return the exact downward speed for a fall from rest with quadratic drag.

    For ``m dv/dt = m g - k v^2`` (downward positive) the exact solution is
    ``v(t) = v_terminal tanh(g t / v_terminal)`` with
    ``v_terminal = sqrt(m g / k)``.  The result is a nonnegative speed.
    """
    t_array = _time_array(t)
    v_terminal = as_positive_float(v_terminal, "v_terminal")
    g = as_positive_float(g, "g")
    if t_array.ndim == 0:
        return v_terminal * math.tanh(g * float(t_array) / v_terminal)
    return v_terminal * np.tanh(g * t_array / v_terminal)


def _log_cosh(x: float) -> float:
    """Evaluate ``ln(cosh(x))`` accurately for both small and large ``x``.

    Two exact identities, chosen by the size of ``x``:

    * small ``|x|``: ``ln(cosh x) = -ln(1 - tanh^2(x)) / 2``.  The direct
      ``ln(cosh(x))`` form loses the short-time ballistic limit
      ``d ~ g t^2 / 2`` to cancellation once ``ln(cosh x)`` underflows
      relative to its own rounding; the ``tanh`` identity keeps it.
    * large ``|x|``: ``ln(cosh x) = logaddexp(x, -x) - ln 2``.  The ``tanh``
      identity loses precision instead once ``tanh^2(x)`` rounds to 1.
    """
    if abs(x) < 1.0:
        return -0.5 * math.log1p(-math.tanh(x) ** 2)
    return float(np.logaddexp(x, -x)) - math.log(2.0)


def quadratic_drag_fall_distance(
    t: object, v_terminal: float, g: float
) -> float | NDArray[np.float64]:
    """Return the exact fallen distance for a fall from rest with quadratic drag.

    ``d(t) = (v_terminal^2 / g) ln(cosh(g t / v_terminal))`` — the integral of
    :func:`quadratic_drag_fall_speed`.  The logarithm is evaluated by
    :func:`_log_cosh`, which stays accurate for both very short and very long
    falls.
    """
    t_array = _time_array(t)
    v_terminal = as_positive_float(v_terminal, "v_terminal")
    g = as_positive_float(g, "g")
    scale = v_terminal**2 / g
    if t_array.ndim == 0:
        x = g * float(t_array) / v_terminal
        return scale * _log_cosh(x)
    x_array = g * t_array / v_terminal
    small = np.abs(x_array) < 1.0
    tanh_argument = np.where(small, x_array, 0.0)
    large_argument = np.where(small, 0.0, x_array)
    log_cosh = np.where(
        small,
        -0.5 * np.log1p(-np.tanh(tanh_argument) ** 2),
        np.logaddexp(large_argument, -large_argument) - math.log(2.0),
    )
    return np.asarray(scale * log_cosh, dtype=float)