import numpy as np
import pytest

from quasarlab.numerical.integrators import euler_step
from quasarlab.numerical.state import State


def test_state_stores_time_position_and_velocity():
    state = State(0.5, (1.0, 2.0), (3.0, -4.0))
    assert state.time == 0.5
    np.testing.assert_allclose(state.position, [1.0, 2.0])
    np.testing.assert_allclose(state.velocity, [3.0, -4.0])


@pytest.mark.parametrize("time", [-0.1, float("nan"), float("inf")])
def test_state_rejects_invalid_time(time):
    with pytest.raises(ValueError):
        State(time, (0.0, 0.0), (0.0, 0.0))


@pytest.mark.parametrize("position", [(1.0,), (1.0, 2.0, 3.0), "bad", (1.0, float("nan"))])
def test_state_rejects_invalid_position(position):
    with pytest.raises((TypeError, ValueError)):
        State(0.0, position, (0.0, 0.0))


@pytest.mark.parametrize("velocity", [(1.0,), (1.0, 2.0, 3.0), None, (float("inf"), 0.0)])
def test_state_rejects_invalid_velocity(velocity):
    with pytest.raises((TypeError, ValueError)):
        State(0.0, (0.0, 0.0), velocity)


def test_euler_velocity_update_matches_hand_calculation():
    state = State(0.0, (0.0, 0.0), (0.0, 10.0))
    updated = euler_step(state, (0.0, -9.81), 0.1)
    np.testing.assert_allclose(updated.velocity, [0.0, 10.0 - 0.981])


def test_euler_position_update_uses_old_velocity():
    state = State(0.0, (0.0, 0.0), (0.0, 10.0))
    updated = euler_step(state, (0.0, -9.81), 0.1)
    np.testing.assert_allclose(updated.position, [0.0, 1.0])


def test_euler_advances_time():
    state = State(2.0, (0.0, 0.0), (1.0, 1.0))
    updated = euler_step(state, (0.0, 0.0), 0.25)
    assert updated.time == 2.25


def test_zero_acceleration_gives_uniform_motion():
    state = State(0.0, (1.0, 2.0), (3.0, -1.0))
    updated = euler_step(state, (0.0, 0.0), 0.5)
    np.testing.assert_allclose(updated.position, [2.5, 1.5])
    np.testing.assert_allclose(updated.velocity, [3.0, -1.0])


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
def test_euler_rejects_invalid_timestep(dt):
    state = State(0.0, (0.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError):
        euler_step(state, (0.0, 0.0), dt)


@pytest.mark.parametrize("acceleration", [(1.0,), (1.0, 2.0, 3.0), "bad", (1.0, float("nan"))])
def test_euler_rejects_invalid_acceleration(acceleration):
    state = State(0.0, (0.0, 0.0), (0.0, 0.0))
    with pytest.raises((TypeError, ValueError)):
        euler_step(state, acceleration, 0.1)


def test_euler_rejects_non_state_input():
    with pytest.raises(TypeError):
        euler_step("not a state", (0.0, 0.0), 0.1)


def test_euler_does_not_mutate_input_state():
    state = State(0.0, (1.0, 2.0), (3.0, 4.0))
    updated = euler_step(state, (1.0, -1.0), 0.1)
    np.testing.assert_allclose(state.position, [1.0, 2.0])
    np.testing.assert_allclose(state.velocity, [3.0, 4.0])
    assert state.time == 0.0
    np.testing.assert_allclose(updated.position, [1.3, 2.4])


def test_constant_acceleration_matches_closed_form():
    x0 = np.array([0.0, 0.0])
    v0 = np.array([2.0, 5.0])
    a = np.array([0.0, -9.81])
    dt = 0.1
    n_steps = 10
    state = State(0.0, x0, v0)
    for _ in range(n_steps):
        state = euler_step(state, a, dt)
    expected_position = x0 + v0 * n_steps * dt + 0.5 * a * n_steps * (n_steps - 1) * dt**2
    expected_velocity = v0 + a * n_steps * dt
    np.testing.assert_allclose(state.position, expected_position, atol=1e-12)
    np.testing.assert_allclose(state.velocity, expected_velocity, atol=1e-12)

    exact_position = x0 + v0 * n_steps * dt + 0.5 * a * (n_steps * dt) ** 2
    expected_error = -0.5 * a * n_steps * dt**2
    np.testing.assert_allclose(state.position - exact_position, expected_error, atol=1e-12)


def test_euler_error_decreases_as_timestep_decreases():
    x0 = np.array([0.0, 0.0])
    v0 = np.array([2.0, 5.0])
    a = np.array([0.0, -9.81])
    errors = []
    for dt in (0.1, 0.01, 0.001):
        n_steps = round(1.0 / dt)
        state = State(0.0, x0, v0)
        for _ in range(n_steps):
            state = euler_step(state, a, dt)
        exact_position = x0 + v0 * 1.0 + 0.5 * a * 1.0**2
        errors.append(np.linalg.norm(state.position - exact_position))
    assert errors[0] > errors[1] > errors[2]
