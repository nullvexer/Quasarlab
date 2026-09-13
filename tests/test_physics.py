import numpy as np
import pytest

from quasarlab.numerical.state import State
from quasarlab.physics import (
    Particle,
    ParticleSystem,
    UniformGravity,
    analytical,
    gravitational_force,
)


def make_state(position=(0.0, 0.0), velocity=(0.0, 0.0), time=0.0):
    return State(time, position, velocity)


class ConstantForce:
    """A test-only force law with a fixed force vector."""

    def __init__(self, force):
        self.force_value = np.asarray(force, dtype=float)

    def force(self, particle, state):
        return self.force_value.copy()


def test_particle_stores_mass_position_and_velocity():
    particle = Particle(2.0, (1.0, -2.0), (0.5, 3.0))
    assert particle.mass == 2.0
    np.testing.assert_allclose(particle.position, [1.0, -2.0])
    np.testing.assert_allclose(particle.velocity, [0.5, 3.0])


@pytest.mark.parametrize("mass", [0.0, -1.0, float("nan"), float("inf")])
def test_particle_rejects_invalid_mass(mass):
    with pytest.raises(ValueError):
        Particle(mass, (0.0, 0.0), (0.0, 0.0))


@pytest.mark.parametrize(
    "position",
    [(1.0,), (1.0, 2.0, 3.0), [[1.0, 2.0]], ["a", "b"], (1.0, float("nan"))],
)
def test_particle_rejects_invalid_position(position):
    with pytest.raises((TypeError, ValueError)):
        Particle(1.0, position, (0.0, 0.0))


def test_gravitational_force_is_mass_times_g():
    gravity = np.array([0.0, -9.81])
    np.testing.assert_allclose(gravitational_force(3.0, gravity), [0.0, -29.43])


def test_uniform_gravity_default_points_downward():
    model = UniformGravity()
    np.testing.assert_allclose(model.g, [0.0, -9.81])


def test_gravitational_force_is_independent_of_state():
    particle = Particle(2.0, (10.0, -5.0), (3.0, 4.0))
    model = UniformGravity(np.array([0.0, -9.81]))
    first = model.force(particle, make_state())
    second = model.force(particle, make_state(position=(100.0, 100.0), velocity=(-1.0, 7.0)))
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first, [0.0, -19.62])


def test_gravity_acceleration_equals_g_and_is_mass_independent():
    model = UniformGravity(np.array([0.0, -9.81]))
    light = Particle(0.1, (0.0, 0.0), (0.0, 0.0))
    heavy = Particle(1000.0, (0.0, 0.0), (0.0, 0.0))
    np.testing.assert_allclose(model.acceleration(light, make_state()), [0.0, -9.81])
    np.testing.assert_allclose(model.acceleration(heavy, make_state()), [0.0, -9.81])


def test_gravity_acceleration_returns_a_copy():
    model = UniformGravity(np.array([0.0, -9.81]))
    acceleration = model.acceleration(Particle(1.0, (0, 0), (0, 0)), make_state())
    acceleration[0] = 99.0
    np.testing.assert_allclose(model.g, [0.0, -9.81])


def test_newtons_second_law_gives_a_equals_f_over_m():
    particle = Particle(2.5, (0.0, 0.0), (0.0, 0.0))
    system = ParticleSystem(particle, [ConstantForce([3.0, 4.0])])
    np.testing.assert_allclose(system.net_force(make_state()), [3.0, 4.0])
    np.testing.assert_allclose(system.acceleration(make_state()), [1.2, 1.6])


def test_force_laws_sum_in_the_system():
    particle = Particle(2.0, (0.0, 0.0), (0.0, 0.0))
    system = ParticleSystem(
        particle,
        [ConstantForce([1.0, 0.0]), UniformGravity(np.array([0.0, -3.0]))],
    )
    np.testing.assert_allclose(system.net_force(make_state()), [1.0, -6.0])


def test_system_without_forces_has_zero_force_and_acceleration():
    particle = Particle(3.0, (0.0, 0.0), (0.0, 0.0))
    system = ParticleSystem(particle, [])
    np.testing.assert_allclose(system.net_force(make_state()), [0.0, 0.0])
    np.testing.assert_allclose(system.acceleration(make_state()), [0.0, 0.0])


def test_system_rejects_invalid_force_law():
    particle = Particle(1.0, (0.0, 0.0), (0.0, 0.0))
    with pytest.raises(TypeError):
        ParticleSystem(particle, [object()])


def test_analytical_position_scalar_time():
    x0 = np.array([1.0, 2.0])
    v0 = np.array([3.0, -1.0])
    a = np.array([0.0, -9.81])
    result = analytical.position(2.0, x0, v0, a)
    expected = x0 + v0 * 2.0 + 0.5 * a * 4.0
    np.testing.assert_allclose(result, expected)


def test_analytical_position_array_time():
    x0 = np.array([0.0, 0.0])
    v0 = np.array([1.0, 2.0])
    a = np.array([0.0, -9.81])
    times = np.array([0.0, 1.0, 2.0])
    result = analytical.position(times, x0, v0, a)
    assert result.shape == (3, 2)
    for index, time in enumerate(times):
        expected = x0 + v0 * time + 0.5 * a * time**2
        np.testing.assert_allclose(result[index], expected)


def test_analytical_velocity_scalar_time():
    v0 = np.array([3.0, -1.0])
    a = np.array([0.0, -9.81])
    np.testing.assert_allclose(analytical.velocity(2.0, v0, a), [3.0, -20.62])


def test_analytical_velocity_array_time():
    v0 = np.array([3.0, -1.0])
    a = np.array([0.0, -9.81])
    times = np.array([0.0, 1.0, 2.0])
    result = analytical.velocity(times, v0, a)
    assert result.shape == (3, 2)
    for index, time in enumerate(times):
        np.testing.assert_allclose(result[index], v0 + a * time)


@pytest.mark.parametrize("time", ["bad", float("nan"), np.array([[1.0, 2.0]])])
def test_analytical_rejects_invalid_time(time):
    with pytest.raises(ValueError):
        analytical.position(time, (0.0, 0.0), (0.0, 0.0), (0.0, -9.81))
