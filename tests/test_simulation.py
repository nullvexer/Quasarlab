import matplotlib.pyplot as plt
import numpy as np
import pytest

from quasarlab.numerical.integrators import euler_step
from quasarlab.numerical.state import State
from quasarlab.physics import Particle, ParticleSystem, UniformGravity, analytical
from quasarlab.simulation.world import Trajectory, World
from quasarlab.visualization.plotting import plot_trajectory


def make_particle(mass=1.0, position=(0.0, 0.0), velocity=(0.0, 0.0)):
    return Particle(mass, position, velocity)


def make_world(mass=1.0, position=(0.0, 0.0), velocity=(0.0, 0.0), dt=0.1, gravity=(0.0, -9.81)):
    particle = make_particle(mass=mass, position=position, velocity=velocity)
    system = ParticleSystem(particle, [UniformGravity(np.asarray(gravity))])
    return World(system, dt=dt)


def test_time_advances_in_equal_steps():
    world = make_world(dt=0.1)
    trajectory = world.run(1.0)
    assert len(trajectory) == 11
    np.testing.assert_allclose(trajectory.time, np.linspace(0.0, 1.0, 11), atol=1e-12)


def test_trajectory_shapes_are_valid():
    trajectory = make_world(dt=0.1).run(1.0)
    assert trajectory.time.shape == (11,)
    assert trajectory.position.shape == (11, 2)
    assert trajectory.velocity.shape == (11, 2)
    assert trajectory.x.shape == (11,)
    assert trajectory.vx.shape == (11,)


def test_trajectory_starts_from_initial_conditions():
    world = make_world(position=(1.0, 2.0), velocity=(3.0, 4.0), dt=0.1)
    trajectory = world.run(0.5)
    np.testing.assert_allclose(trajectory.position[0], [1.0, 2.0])
    np.testing.assert_allclose(trajectory.velocity[0], [3.0, 4.0])
    assert trajectory.time[0] == 0.0


def test_gravity_acts_downward_and_horizontal_velocity_is_constant():
    world = make_world(velocity=(2.0, 5.0), dt=0.1)
    trajectory = world.run(1.0)
    np.testing.assert_allclose(trajectory.vx, 2.0)
    assert np.all(np.diff(trajectory.vy) < 0.0)
    assert trajectory.vy[-1] < trajectory.vy[0]


def test_zero_initial_velocity_falls_straight_down():
    world = make_world(position=(3.0, 10.0), velocity=(0.0, 0.0), dt=0.1)
    trajectory = world.run(1.0)
    np.testing.assert_allclose(trajectory.x, 3.0)
    assert np.all(np.diff(trajectory.y) <= 0.0)
    assert np.all(np.diff(trajectory.y[1:]) < 0.0)


def test_zero_duration_returns_initial_sample():
    world = make_world(position=(1.0, 2.0), velocity=(3.0, 4.0), dt=0.1)
    trajectory = world.run(0.0)
    assert len(trajectory) == 1
    np.testing.assert_allclose(trajectory.position[0], [1.0, 2.0])
    np.testing.assert_allclose(trajectory.velocity[0], [3.0, 4.0])


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
def test_world_rejects_invalid_timestep(dt):
    with pytest.raises(ValueError):
        make_world(dt=dt)


@pytest.mark.parametrize("duration", [-1.0, 0.15, float("nan")])
def test_world_rejects_invalid_duration(duration):
    with pytest.raises(ValueError):
        make_world(dt=0.1).run(duration)


@pytest.mark.parametrize("n_steps", [-1, 1.5, True])
def test_world_rejects_invalid_step_count(n_steps):
    with pytest.raises(ValueError):
        make_world(dt=0.1).run_steps(n_steps)


def test_world_rejects_non_particle_system():
    with pytest.raises(TypeError):
        World("not a system", dt=0.1)


def test_same_inputs_give_identical_results():
    world = make_world(position=(1.0, 2.0), velocity=(3.0, 4.0), dt=0.01)
    first = world.run(1.0)
    second = world.run(1.0)
    np.testing.assert_array_equal(first.time, second.time)
    np.testing.assert_array_equal(first.position, second.position)
    np.testing.assert_array_equal(first.velocity, second.velocity)


def test_uniform_gravity_trajectory_is_independent_of_mass():
    first = make_world(mass=0.5, velocity=(2.0, 5.0), dt=0.01).run(1.0)
    second = make_world(mass=5.0, velocity=(2.0, 5.0), dt=0.01).run(1.0)
    np.testing.assert_allclose(first.position, second.position, atol=1e-12)
    np.testing.assert_allclose(first.velocity, second.velocity, atol=1e-12)


def test_trajectory_matches_manual_euler_stepping():
    gravity = np.array([0.0, -9.81])
    world = make_world(velocity=(2.0, 5.0), dt=0.1, gravity=gravity)
    trajectory = world.run(1.0)

    state = State(0.0, (0.0, 0.0), (2.0, 5.0))
    positions = [state.position.copy()]
    velocities = [state.velocity.copy()]
    times = [state.time]
    for _ in range(10):
        state = euler_step(state, gravity, 0.1)
        times.append(state.time)
        positions.append(state.position.copy())
        velocities.append(state.velocity.copy())

    np.testing.assert_allclose(trajectory.time, times)
    np.testing.assert_allclose(trajectory.position, positions)
    np.testing.assert_allclose(trajectory.velocity, velocities)


def test_numerical_solution_converges_to_analytical_solution():
    gravity = np.array([0.0, -9.81])
    errors = []
    for dt in (0.1, 0.01, 0.001):
        world = make_world(velocity=(2.0, 5.0), dt=dt, gravity=gravity)
        trajectory = world.run(1.0)
        reference = analytical.position(trajectory.time, (0.0, 0.0), (2.0, 5.0), gravity)
        errors.append(np.max(np.linalg.norm(trajectory.position - reference, axis=1)))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-2


def test_trajectory_rejects_mismatched_arrays():
    with pytest.raises(ValueError):
        Trajectory(np.array([0.0, 0.1]), np.zeros((3, 2)), np.zeros((2, 2)))


def test_plot_trajectory_saves_file(tmp_path):
    world = make_world(velocity=(2.0, 5.0), dt=0.1)
    trajectory = world.run(1.0)
    reference = analytical.position(trajectory.time, (0.0, 0.0), (2.0, 5.0), (0.0, -9.81))
    save_path = tmp_path / "trajectory.png"
    fig, _ax = plot_trajectory(trajectory, analytical_positions=reference, save_path=save_path)
    try:
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    finally:
        plt.close(fig)
