"""Numerical integration algorithms.

V0.1 implements explicit Euler integration:

    v_{n+1} = v_n + a_n dt
    x_{n+1} = x_n + v_n dt

The position update deliberately uses the old velocity; that is the defining
property of explicit Euler.
"""

from __future__ import annotations

from quasarlab._validation import as_positive_float, as_vector
from quasarlab.numerical.state import State


def euler_step(state: State, acceleration: object, dt: float) -> State:
    """Advance ``state`` by one explicit Euler step of size ``dt`` seconds.

    ``acceleration`` is the acceleration vector in m/s^2 at the current state.
    The input state is not modified.
    """
    if not isinstance(state, State):
        raise TypeError(f"state must be a State, got {type(state).__name__}")
    dt = as_positive_float(dt, "dt")
    a = as_vector(acceleration, "acceleration")
    new_velocity = state.velocity + a * dt
    new_position = state.position + state.velocity * dt
    return State(state.time + dt, new_position, new_velocity)
