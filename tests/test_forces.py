import numpy as np
import pytest

from quasarlab.numerical.state import State
from quasarlab.physics.forces import ConstantForce, LinearDrag, QuadraticDrag, UniformGravity
from quasarlab.physics.particle import Particle
from quasarlab.physics.systems import ParticleSystem
from quasarlab.simulation.world import World


def make_particle(mass=1.0, position=(0.0, 0.0), velocity=(0.0, 0.0)):
    return Particle(mass, position, velocity)


def make_state(position=(0.0, 0.0), velocity=(0.0, 0.0), time=0.0):
    return State(time, position, velocity)


class TestConstantForce:
    def test_force_is_the_constant_vector(self):
        law = ConstantForce((3.0, -4.0))
        for state in (
            make_state(),
            make_state(position=(10.0, -2.0), velocity=(-5.0, 7.0), time=3.0),
        ):
            np.testing.assert_allclose(law.force(make_particle(), state), [3.0, -4.0])

    def test_force_is_independent_of_mass(self):
        law = ConstantForce((3.0, -4.0))
        np.testing.assert_allclose(
            law.force(make_particle(mass=0.1), make_state()), [3.0, -4.0]
        )
        np.testing.assert_allclose(
            law.force(make_particle(mass=100.0), make_state()), [3.0, -4.0]
        )

    def test_rejects_malformed_force(self):
        for bad in [(1.0,), (1.0, 2.0, 3.0), "bad", (1.0, float("nan"))]:
            with pytest.raises((TypeError, ValueError)):
                ConstantForce(bad)

    def test_stores_a_copy_of_the_force_vector(self):
        vector = np.array([1.0, 2.0])
        law = ConstantForce(vector)
        vector[0] = 99.0
        np.testing.assert_allclose(law.force_value, [1.0, 2.0])

    def test_reversing_the_force_reverses_the_contribution(self):
        state = make_state(velocity=(1.0, 2.0))
        forward = ConstantForce((5.0, -1.0)).force(make_particle(), state)
        backward = ConstantForce((-5.0, 1.0)).force(make_particle(), state)
        np.testing.assert_allclose(forward, -backward)

    def test_zero_force_law_contributes_nothing(self):
        particle = make_particle(velocity=(10.0, 5.0))
        with_gravity = ParticleSystem(particle, [UniformGravity()])
        with_zero = ParticleSystem(particle, [UniformGravity(), ConstantForce((0.0, 0.0))])
        state = make_state()
        np.testing.assert_array_equal(
            with_gravity.net_force(state), with_zero.net_force(state)
        )

    def test_default_label(self):
        assert ConstantForce((1.0, 0.0)).label == "applied force"

    def test_custom_label(self):
        assert ConstantForce((1.0, 0.0), label="push").label == "push"


class TestLinearDrag:
    def test_force_is_minus_b_times_velocity(self):
        law = LinearDrag(0.5)
        particle = make_particle()
        for velocity in ((3.0, -4.0), (-2.0, 0.5), (0.0, -7.0)):
            state = make_state(velocity=velocity)
            np.testing.assert_allclose(law.force(particle, state), -0.5 * np.asarray(velocity))

    def test_rejects_negative_drag_coefficient(self):
        with pytest.raises(ValueError):
            LinearDrag(-0.1)

    def test_accepts_zero_drag_coefficient(self):
        law = LinearDrag(0.0)
        np.testing.assert_allclose(
            law.force(make_particle(), make_state(velocity=(3.0, 4.0))), [0.0, 0.0]
        )

    def test_zero_velocity_gives_zero_force(self):
        law = LinearDrag(2.0)
        force = law.force(make_particle(), make_state())
        np.testing.assert_array_equal(force, [0.0, 0.0])
        assert np.all(np.isfinite(force))

    def test_reversing_velocity_reverses_the_force(self):
        law = LinearDrag(0.7)
        particle = make_particle()
        forward = law.force(particle, make_state(velocity=(2.0, -1.0)))
        backward = law.force(particle, make_state(velocity=(-2.0, 1.0)))
        np.testing.assert_allclose(forward, -backward)

    def test_drag_always_opposes_velocity(self):
        law = LinearDrag(1.3)
        particle = make_particle()
        for velocity in ((3.0, 1.0), (-4.0, 2.0), (0.0, -5.0), (1.0, -1.0)):
            force = law.force(particle, make_state(velocity=velocity))
            assert float(np.dot(force, velocity)) < 0.0

    def test_force_magnitude_scales_linearly_with_b(self):
        velocity = (2.0, -3.0)
        weak = LinearDrag(0.4).force(make_particle(), make_state(velocity=velocity))
        strong = LinearDrag(1.2).force(make_particle(), make_state(velocity=velocity))
        np.testing.assert_allclose(strong, 3.0 * weak)

    def test_force_magnitude_is_rotation_invariant(self):
        law = LinearDrag(0.9)
        speed = 5.0
        first = law.force(make_particle(), make_state(velocity=(speed, 0.0)))
        angle = np.radians(37.0)
        rotated = (speed * np.cos(angle), speed * np.sin(angle))
        second = law.force(make_particle(), make_state(velocity=rotated))
        assert float(np.linalg.norm(first)) == pytest.approx(float(np.linalg.norm(second)))

    def test_force_is_independent_of_mass(self):
        law = LinearDrag(0.5)
        state = make_state(velocity=(4.0, -2.0))
        np.testing.assert_allclose(
            law.force(make_particle(mass=0.2), state),
            law.force(make_particle(mass=80.0), state),
        )

    def test_default_label(self):
        assert LinearDrag(0.1).label == "linear drag"


class TestQuadraticDrag:
    def make_law(self, density=2.0, drag_coefficient=1.0, area=0.5):
        return QuadraticDrag(density, drag_coefficient, area)

    def test_force_is_minus_half_rho_cd_a_speed_times_velocity(self):
        law = self.make_law()
        particle = make_particle()
        velocity = np.array([3.0, -4.0])
        force = law.force(particle, make_state(velocity=velocity))
        k = 0.5 * 2.0 * 1.0 * 0.5
        np.testing.assert_allclose(force, -k * float(np.linalg.norm(velocity)) * velocity)

    def test_rejects_negative_parameters(self):
        for kwargs in (
            {"density": -1.0, "drag_coefficient": 1.0, "area": 0.5},
            {"density": 2.0, "drag_coefficient": -0.5, "area": 0.5},
            {"density": 2.0, "drag_coefficient": 1.0, "area": -0.2},
        ):
            with pytest.raises(ValueError):
                QuadraticDrag(**kwargs)

    def test_accepts_zero_parameters(self):
        for kwargs in (
            {"density": 0.0, "drag_coefficient": 1.0, "area": 0.5},
            {"density": 2.0, "drag_coefficient": 0.0, "area": 0.5},
            {"density": 2.0, "drag_coefficient": 1.0, "area": 0.0},
        ):
            law = QuadraticDrag(**kwargs)
            np.testing.assert_allclose(
                law.force(make_particle(), make_state(velocity=(3.0, 4.0))), [0.0, 0.0]
            )

    def test_zero_velocity_gives_zero_force_without_nan(self):
        law = self.make_law()
        force = law.force(make_particle(), make_state())
        np.testing.assert_array_equal(force, [0.0, 0.0])
        assert np.all(np.isfinite(force))

    def test_drag_opposes_velocity_for_all_directions(self):
        law = self.make_law()
        particle = make_particle()
        for velocity in ((3.0, 1.0), (-4.0, 2.0), (0.0, -5.0), (-1.0, -1.0)):
            force = law.force(particle, make_state(velocity=velocity))
            assert float(np.dot(force, velocity)) < 0.0

    def test_reversing_velocity_reverses_the_force(self):
        law = self.make_law()
        particle = make_particle()
        forward = law.force(particle, make_state(velocity=(2.0, -1.0)))
        backward = law.force(particle, make_state(velocity=(-2.0, 1.0)))
        np.testing.assert_allclose(forward, -backward)

    def test_force_magnitude_grows_quadratically_with_speed(self):
        law = self.make_law()
        particle = make_particle()
        slow = float(
            np.linalg.norm(law.force(particle, make_state(velocity=(1.0, 0.0))))
        )
        fast = float(
            np.linalg.norm(law.force(particle, make_state(velocity=(2.0, 0.0))))
        )
        assert fast == pytest.approx(4.0 * slow)

    def test_force_magnitude_is_rotation_invariant(self):
        law = self.make_law()
        particle = make_particle()
        speed = 6.0
        first = float(np.linalg.norm(law.force(particle, make_state(velocity=(speed, 0.0)))))
        angle = np.radians(112.0)
        rotated = (speed * np.cos(angle), speed * np.sin(angle))
        second = float(np.linalg.norm(law.force(particle, make_state(velocity=rotated))))
        assert second == pytest.approx(first)

    def test_each_parameter_takes_the_law_to_zero_individually(self):
        particle = make_particle()
        state = make_state(velocity=(3.0, -4.0))
        for kwargs in (
            {"density": 0.0, "drag_coefficient": 1.0, "area": 0.5},
            {"density": 2.0, "drag_coefficient": 0.0, "area": 0.5},
            {"density": 2.0, "drag_coefficient": 1.0, "area": 0.0},
        ):
            law = QuadraticDrag(**kwargs)
            np.testing.assert_allclose(law.force(particle, state), [0.0, 0.0])

    def test_terminal_speed_formula(self):
        law = self.make_law()
        mass = 2.0
        gravity = 9.81
        expected = np.sqrt(2.0 * mass * gravity / (2.0 * 1.0 * 0.5))
        assert law.terminal_speed(mass, gravity) == pytest.approx(expected)

    def test_terminal_speed_rejects_zero_drag(self):
        law = QuadraticDrag(2.0, 1.0, 0.0)
        with pytest.raises(ValueError):
            law.terminal_speed(1.0, 9.81)

    def test_terminal_speed_rejects_invalid_input(self):
        law = self.make_law()
        with pytest.raises(ValueError):
            law.terminal_speed(0.0, 9.81)
        with pytest.raises(ValueError):
            law.terminal_speed(1.0, -9.81)

    def test_default_label(self):
        assert self.make_law().label == "quadratic drag"


class TestForceComposition:
    def make_system(self, mass=2.0, velocity=(6.0, -3.0)):
        particle = make_particle(mass=mass, velocity=velocity)
        system = ParticleSystem(
            particle,
            [
                UniformGravity((0.0, -9.81)),
                ConstantForce((4.0, 1.0)),
                LinearDrag(0.3),
                QuadraticDrag(1.0, 1.0, 0.5),
            ],
        )
        return particle, system

    def test_net_force_equals_independent_sum_of_contributions(self):
        particle, system = self.make_system()
        state = make_state(velocity=(6.0, -3.0))
        expected = (
            UniformGravity((0.0, -9.81)).force(particle, state)
            + ConstantForce((4.0, 1.0)).force(particle, state)
            + LinearDrag(0.3).force(particle, state)
            + QuadraticDrag(1.0, 1.0, 0.5).force(particle, state)
        )
        np.testing.assert_allclose(system.net_force(state), expected)

    def test_force_contributions_labels_and_values(self):
        _, system = self.make_system()
        state = make_state(velocity=(6.0, -3.0))
        contributions = system.force_contributions(state)
        labels = [name for name, _ in contributions]
        assert labels == [
            "uniform gravity",
            "applied force",
            "linear drag",
            "quadratic drag",
        ]
        total = np.zeros(2)
        for _, force in contributions:
            total = total + force
        np.testing.assert_allclose(total, system.net_force(state))

    def test_contributions_work_for_user_laws_without_labels(self):
        class Anonymous:
            def force(self, particle, state):
                return np.array([1.0, 0.0])

        system = ParticleSystem(make_particle(), [Anonymous()])
        contributions = system.force_contributions(make_state())
        assert contributions[0][0] == "Anonymous"
        np.testing.assert_allclose(contributions[0][1], [1.0, 0.0])

    def test_zero_force_law_does_not_change_the_trajectory(self):
        particle = make_particle(velocity=(10.0, 5.0))
        plain = World(ParticleSystem(particle, [UniformGravity()]), dt=0.01).run(1.0)
        particle_copy = make_particle(velocity=(10.0, 5.0))
        with_zero = World(
            ParticleSystem(particle_copy, [UniformGravity(), ConstantForce((0.0, 0.0))]),
            dt=0.01,
        ).run(1.0)
        np.testing.assert_array_equal(plain.position, with_zero.position)
        np.testing.assert_array_equal(plain.velocity, with_zero.velocity)

    def test_gravity_trajectory_is_mass_independent(self):
        first = World(
            ParticleSystem(make_particle(mass=0.5, velocity=(3.0, 4.0)), [UniformGravity()]),
            dt=0.01,
        ).run(1.0)
        second = World(
            ParticleSystem(make_particle(mass=50.0, velocity=(3.0, 4.0)), [UniformGravity()]),
            dt=0.01,
        ).run(1.0)
        np.testing.assert_allclose(first.position, second.position, atol=1e-12)

    def test_drag_trajectory_is_mass_dependent(self):
        first = World(
            ParticleSystem(
                make_particle(mass=0.5, velocity=(3.0, 4.0)),
                [UniformGravity(), LinearDrag(0.4)],
            ),
            dt=0.01,
        ).run(1.0)
        second = World(
            ParticleSystem(
                make_particle(mass=50.0, velocity=(3.0, 4.0)),
                [UniformGravity(), LinearDrag(0.4)],
            ),
            dt=0.01,
        ).run(1.0)
        assert not np.allclose(first.position, second.position, atol=1e-6)

    def test_composed_scenario_matches_newton_second_law(self):
        _, system = self.make_system(mass=2.0)
        state = make_state(velocity=(6.0, -3.0))
        expected_acceleration = system.net_force(state) / 2.0
        np.testing.assert_allclose(system.acceleration(state), expected_acceleration)
