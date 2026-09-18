import numpy as np
import pytest

from quasarlab.numerical.state import State
from quasarlab.physics import UniformGravity
from quasarlab.physics.analytical import (
    linear_drag_position,
    linear_drag_velocity,
    quadratic_drag_fall_distance,
    quadratic_drag_fall_speed,
)
from quasarlab.physics.forces import ConstantForce, LinearDrag, QuadraticDrag
from quasarlab.physics.particle import Particle
from quasarlab.physics.systems import ParticleSystem
from quasarlab.simulation.world import World

GRAVITY = 9.81
MASS = 2.0
DRAG_B = 0.4
GAMMA = DRAG_B / MASS
V_TERMINAL_LINEAR = -GRAVITY / GAMMA

QUADRATIC_K = 0.5 * 2.0 * 1.0 * 0.5
V_TERMINAL_QUADRATIC = np.sqrt(MASS * GRAVITY / QUADRATIC_K)


def make_world(*laws, mass=MASS, position=(0.0, 0.0), velocity=(0.0, 0.0), dt=0.001):
    particle = Particle(mass, position, velocity)
    return World(ParticleSystem(particle, laws), dt=dt)


def sample_at(result, time, dt):
    return round(time / dt)


class TestLinearDragExactValidation:
    def make_fall_world(self, dt=0.001):
        return make_world(
            UniformGravity((0.0, -GRAVITY)),
            LinearDrag(DRAG_B),
            position=(0.0, 20.0),
            dt=dt,
        )

    def test_velocity_matches_exact_exponential_solution(self):
        result = self.make_fall_world().run(3.0)
        for time in (0.5, 1.5, 3.0):
            exact = linear_drag_velocity(time, 0.0, GAMMA, V_TERMINAL_LINEAR)
            numerical = result.vy[sample_at(result, time, 0.001)]
            assert numerical == pytest.approx(exact, abs=0.01)

    def test_position_matches_exact_exponential_solution(self):
        result = self.make_fall_world().run(3.0)
        for time in (0.5, 1.5, 3.0):
            exact = linear_drag_position(time, 20.0, 0.0, GAMMA, V_TERMINAL_LINEAR)
            numerical = result.y[sample_at(result, time, 0.001)]
            assert numerical == pytest.approx(exact, abs=0.02)

    def test_horizontal_push_with_drag_matches_full_2d_exact_solution(self):
        # Linear drag decouples per component: vx obeys the drag ODE with
        # driving force F_x, vy obeys it with driving force F_y = -m g.
        result = make_world(
            UniformGravity((0.0, -GRAVITY)),
            ConstantForce((4.0, 0.0)),
            LinearDrag(DRAG_B),
            dt=0.001,
        ).run(3.0)
        v_term_x = 4.0 / DRAG_B
        v_term_y = -MASS * GRAVITY / DRAG_B
        for time in (1.0, 2.0, 3.0):
            index = sample_at(result, time, 0.001)
            exact_vx = linear_drag_velocity(time, 0.0, GAMMA, v_term_x)
            exact_x = linear_drag_position(time, 0.0, 0.0, GAMMA, v_term_x)
            exact_vy = linear_drag_velocity(time, 0.0, GAMMA, v_term_y)
            exact_y = linear_drag_position(time, 0.0, 0.0, GAMMA, v_term_y)
            assert result.vx[index] == pytest.approx(exact_vx, abs=0.01)
            assert result.x[index] == pytest.approx(exact_x, abs=0.02)
            assert result.vy[index] == pytest.approx(exact_vy, abs=0.01)
            assert result.y[index] == pytest.approx(exact_y, abs=0.02)

    def test_terminal_velocity_is_reached_from_rest(self):
        result = self.make_fall_world(dt=0.01).run(40.0)
        assert abs(result.vy[-1] - V_TERMINAL_LINEAR) < 0.2
        increasing_speed = np.diff(result.vy[: len(result) // 2])
        assert np.all(increasing_speed < 0.0)

    def test_speed_approaches_terminal_from_above(self):
        result = make_world(
            UniformGravity((0.0, -GRAVITY)),
            LinearDrag(DRAG_B),
            velocity=(0.0, -60.0),
            dt=0.01,
        ).run(20.0)
        speed = -result.vy
        assert np.all(np.diff(speed) < 0.0)
        assert abs(speed[-1] - abs(V_TERMINAL_LINEAR)) < 0.5

    def test_error_decreases_with_timestep(self):
        errors = []
        for dt in (0.1, 0.01, 0.001):
            result = self.make_fall_world(dt=dt).run(3.0)
            times = result.time
            exact = linear_drag_velocity(times, 0.0, GAMMA, V_TERMINAL_LINEAR)
            errors.append(float(np.max(np.abs(result.vy - exact))))
        assert errors[0] > errors[1] > errors[2]
        assert 3.0 < errors[0] / errors[1] < 30.0
        assert 3.0 < errors[1] / errors[2] < 30.0

    def test_zero_linear_drag_reduces_exactly_to_ballistic(self):
        with_drag = make_world(UniformGravity((0.0, -GRAVITY)), LinearDrag(0.0)).run(1.0)
        without_drag = make_world(UniformGravity((0.0, -GRAVITY))).run(1.0)
        np.testing.assert_array_equal(with_drag.position, without_drag.position)
        np.testing.assert_array_equal(with_drag.velocity, without_drag.velocity)


class TestQuadraticDragValidation:
    def make_fall_world(self, dt=0.0005, velocity=(0.0, 0.0), height=50.0):
        return make_world(
            UniformGravity((0.0, -GRAVITY)),
            QuadraticDrag(2.0, 1.0, 0.5),
            mass=MASS,
            position=(0.0, height),
            velocity=velocity,
            dt=dt,
        )

    def test_fall_speed_matches_exact_tanh_solution(self):
        result = self.make_fall_world().run(3.0)
        for time in (1.0, 2.0, 3.0):
            exact = quadratic_drag_fall_speed(time, V_TERMINAL_QUADRATIC, GRAVITY)
            numerical = -result.vy[sample_at(result, time, 0.0005)]
            assert numerical == pytest.approx(exact, abs=0.02)

    def test_fall_distance_matches_exact_log_cosh_solution(self):
        result = self.make_fall_world().run(3.0)
        for time in (1.0, 2.0, 3.0):
            fallen = quadratic_drag_fall_distance(time, V_TERMINAL_QUADRATIC, GRAVITY)
            numerical = 50.0 - result.y[sample_at(result, time, 0.0005)]
            assert numerical == pytest.approx(fallen, abs=0.05)

    def test_speed_approaches_terminal_from_above(self):
        result = self.make_fall_world(dt=0.01, velocity=(0.0, -15.0)).run(30.0)
        speed = -result.vy
        # Monotone non-increasing: the tail saturates exactly at Euler's
        # discrete fixed point, which matches v_t to floating-point precision.
        assert np.all(np.diff(speed) <= 0.0)
        assert speed[0] > speed[-1]
        assert abs(speed[-1] - V_TERMINAL_QUADRATIC) < 0.3

    def test_drag_force_at_terminal_speed_balances_weight(self):
        # Independent equilibrium check: k v_t^2 must equal m g.
        law = QuadraticDrag(2.0, 1.0, 0.5)
        particle = Particle(MASS, (0.0, 0.0), (0.0, -V_TERMINAL_QUADRATIC))
        force = law.force(particle, State(0.0, particle.position, particle.velocity))
        np.testing.assert_allclose(force, [0.0, MASS * GRAVITY], atol=1e-9)

    def test_error_decreases_with_timestep(self):
        errors = []
        for dt in (0.005, 0.001, 0.0002):
            result = self.make_fall_world(dt=dt).run(2.0)
            times = result.time
            exact = -quadratic_drag_fall_speed(times, V_TERMINAL_QUADRATIC, GRAVITY)
            errors.append(float(np.max(np.abs(result.vy - exact))))
        assert errors[0] > errors[1] > errors[2]

    def test_zero_quadratic_drag_reduces_exactly_to_ballistic(self):
        with_drag = make_world(UniformGravity((0.0, -GRAVITY)), QuadraticDrag(0.0, 1.0, 1.0))
        without_drag = make_world(UniformGravity((0.0, -GRAVITY)))
        first = with_drag.run(1.0)
        second = without_drag.run(1.0)
        np.testing.assert_array_equal(first.position, second.position)
        np.testing.assert_array_equal(first.velocity, second.velocity)


class TestAnalyticalReferenceFunctions:
    @pytest.mark.parametrize("time", [1e-10, np.array([0.0, 1e-10, 1e-8])])
    def test_quadratic_distance_preserves_short_time_ballistic_limit(self, time):
        distance = quadratic_drag_fall_distance(time, 10.0, 10.0)
        expected = 5.0 * np.asarray(time) ** 2
        np.testing.assert_allclose(distance, expected, rtol=1e-14, atol=0.0)

    def test_quadratic_distance_handles_mixed_time_scales(self):
        times = np.array([1e-10, 1.0, 1000.0])
        expected = np.array([5e-20, 4.337808304830272, 10000.0 - 10.0 * np.log(2.0)])
        with np.errstate(over="raise", invalid="raise"):
            distance = quadratic_drag_fall_distance(times, 10.0, 10.0)
        np.testing.assert_allclose(distance, expected, rtol=1e-14, atol=0.0)

    @pytest.mark.parametrize("time", [1000.0, np.array([1000.0, 2000.0])])
    def test_quadratic_distance_remains_finite_for_long_falls(self, time):
        distance = quadratic_drag_fall_distance(time, 10.0, 10.0)
        expected = 10.0 * np.asarray(time) - 10.0 * np.log(2.0)
        assert np.all(np.isfinite(distance))
        np.testing.assert_allclose(distance, expected, rtol=1e-14)

    def test_linear_drag_velocity_known_values(self):
        # v(t) = v_term + (v0 - v_term) e^{-gamma t}, hand-checked at t=0.
        assert linear_drag_velocity(0.0, 5.0, GAMMA, -10.0) == pytest.approx(5.0)
        exact = -10.0 + 15.0 * np.exp(-GAMMA * 2.0)
        assert linear_drag_velocity(2.0, 5.0, GAMMA, -10.0) == pytest.approx(exact)

    def test_linear_drag_position_known_values(self):
        assert linear_drag_position(0.0, 7.0, 5.0, GAMMA, -10.0) == pytest.approx(7.0)
        exact = 7.0 + (-10.0) * 2.0 + 15.0 * (1.0 - np.exp(-GAMMA * 2.0)) / GAMMA
        assert linear_drag_position(2.0, 7.0, 5.0, GAMMA, -10.0) == pytest.approx(exact)

    def test_linear_drag_references_reject_invalid_input(self):
        with pytest.raises(ValueError):
            linear_drag_velocity(1.0, 0.0, 0.0, -10.0)
        with pytest.raises(ValueError):
            linear_drag_position(1.0, 0.0, 0.0, -1.0, -10.0)
        with pytest.raises(ValueError):
            linear_drag_velocity(float("nan"), 0.0, GAMMA, -10.0)

    def test_quadratic_references_zero_at_rest_and_match_hand_values(self):
        assert quadratic_drag_fall_speed(0.0, V_TERMINAL_QUADRATIC, GRAVITY) == 0.0
        assert quadratic_drag_fall_distance(0.0, V_TERMINAL_QUADRATIC, GRAVITY) == 0.0
        exact_speed = V_TERMINAL_QUADRATIC * np.tanh(
            GRAVITY * 1.0 / V_TERMINAL_QUADRATIC
        )
        assert quadratic_drag_fall_speed(1.0, V_TERMINAL_QUADRATIC, GRAVITY) == pytest.approx(
            exact_speed
        )

    def test_quadratic_references_reject_invalid_input(self):
        with pytest.raises(ValueError):
            quadratic_drag_fall_speed(1.0, 0.0, GRAVITY)
        with pytest.raises(ValueError):
            quadratic_drag_fall_distance(1.0, V_TERMINAL_QUADRATIC, -GRAVITY)
        with pytest.raises(ValueError):
            quadratic_drag_fall_speed(float("inf"), V_TERMINAL_QUADRATIC, GRAVITY)
