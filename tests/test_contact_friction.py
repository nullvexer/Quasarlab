import math

import numpy as np
import pytest

from quasarlab.numerical.state import State
from quasarlab.physics import UniformGravity, analytical
from quasarlab.physics.contact import PlaneSurface, horizontal_surface, inclined_surface
from quasarlab.physics.forces import ConstantForce
from quasarlab.physics.particle import Particle
from quasarlab.physics.systems import ParticleSystem
from quasarlab.simulation.world import World

GRAVITY = 9.81
MASS = 2.0
WEIGHT = MASS * GRAVITY
MU_S = 0.5
MU_K = 0.3
F_S_MAX = MU_S * WEIGHT
F_KINETIC = MU_K * WEIGHT
A_KINETIC = F_KINETIC / MASS


def make_surface_world(
    *laws,
    mu_s=MU_S,
    mu_k=MU_K,
    mass=MASS,
    position=(0.0, 0.0),
    velocity=(0.0, 0.0),
    dt=0.001,
    incline_angle=None,
):
    particle = Particle(mass, position, velocity)
    forces = [UniformGravity((0.0, -GRAVITY)), *laws]
    if incline_angle is None:
        surface = horizontal_surface(mu_s, mu_k)
    else:
        surface = inclined_surface(mu_s, mu_k, incline_angle)
    world = World(ParticleSystem(particle, forces), dt=dt, contact=surface)
    return world, surface, particle


def gravity_force(particle):
    return np.array([0.0, -GRAVITY * particle.mass])


class TestSurfaceConstruction:
    def test_horizontal_surface_geometry(self):
        surface = horizontal_surface(0.5, 0.3)
        np.testing.assert_allclose(surface.normal, [0.0, 1.0])
        np.testing.assert_allclose(surface.tangent, [1.0, 0.0])

    def test_inclined_surface_geometry(self):
        surface = inclined_surface(0.7, 0.5, math.radians(30.0))
        np.testing.assert_allclose(
            surface.normal, [-0.5, math.sqrt(3.0) / 2.0], atol=1e-12
        )
        np.testing.assert_allclose(
            surface.tangent, [math.sqrt(3.0) / 2.0, 0.5], atol=1e-12
        )

    def test_rejects_negative_coefficients(self):
        with pytest.raises(ValueError):
            horizontal_surface(-0.1, 0.3)
        with pytest.raises(ValueError):
            horizontal_surface(0.5, -0.3)

    def test_rejects_kinetic_exceeding_static(self):
        with pytest.raises(ValueError):
            horizontal_surface(0.2, 0.5)

    def test_rejects_zero_normal(self):
        with pytest.raises(ValueError):
            PlaneSurface((0.0, 0.0), (0.0, 0.0), 0.5, 0.3)


class TestNormalForce:
    def test_normal_velocity_is_rejected_before_advancing(self):
        world, _, _ = make_surface_world(velocity=(0.0, 1.0))
        with pytest.raises(ValueError, match="velocity"):
            world.run_steps(1)

    def test_extra_vertical_load_changes_friction(self):
        world, surface, particle = make_surface_world(
            ConstantForce((0.0, -10.0)), velocity=(1.0, 0.0)
        )
        force = surface.reaction(
            particle, world.initial_state, world.system.net_force(world.initial_state)
        )
        np.testing.assert_allclose(force, [-MU_K * (WEIGHT + 10.0), WEIGHT + 10.0])

    def test_zero_step_run_validates_contact(self):
        world, _, _ = make_surface_world(position=(0.0, 1.0))
        with pytest.raises(ValueError, match="surface"):
            world.run_steps(0)

    def test_normal_force_equals_weight_at_rest_on_horizontal_surface(self):
        _, surface, particle = make_surface_world()
        state = State(0.0, (0.0, 0.0), (0.0, 0.0))
        reaction = surface.reaction(particle, state, gravity_force(particle))
        np.testing.assert_allclose(reaction, [0.0, WEIGHT], atol=1e-12)

    def test_incline_normal_is_weight_times_cosine(self):
        angle = math.radians(30.0)
        _, surface, particle = make_surface_world(mu_s=0.7, mu_k=0.5, incline_angle=angle)
        state = State(0.0, (0.0, 0.0), (0.0, 0.0))
        reaction = surface.reaction(particle, state, gravity_force(particle))
        assert float(np.dot(reaction, surface.normal)) == pytest.approx(
            WEIGHT * math.cos(angle), rel=1e-12
        )

    def test_static_equilibrium_on_incline_cancels_gravity_exactly(self):
        angle = math.radians(30.0)
        _, surface, particle = make_surface_world(mu_s=0.7, mu_k=0.5, incline_angle=angle)
        state = State(0.0, (0.0, 0.0), (0.0, 0.0))
        reaction = surface.reaction(particle, state, gravity_force(particle))
        np.testing.assert_allclose(reaction, [0.0, WEIGHT], atol=1e-9)

    def test_static_friction_is_not_the_maximum_value(self):
        # Required friction on the incline is m g sin(theta), which is strictly
        # less than mu_s N: static friction supplies what is needed, not more.
        angle = math.radians(30.0)
        _, surface, particle = make_surface_world(mu_s=0.7, mu_k=0.5, incline_angle=angle)
        state = State(0.0, (0.0, 0.0), (0.0, 0.0))
        reaction = surface.reaction(particle, state, gravity_force(particle))
        required = WEIGHT * math.sin(angle)
        maximum = 0.7 * WEIGHT * math.cos(angle)
        assert float(np.dot(reaction, surface.tangent)) == pytest.approx(required)
        assert required != maximum
        assert abs(float(np.dot(reaction, surface.tangent))) < maximum

    def test_particle_must_be_on_the_surface(self):
        _, surface, particle = make_surface_world()
        state = State(0.0, (0.0, 1.0), (0.0, 0.0))
        with pytest.raises(ValueError):
            surface.reaction(particle, state, gravity_force(particle))

    def test_leaving_the_surface_raises(self):
        _, surface, particle = make_surface_world()
        state = State(0.0, (0.0, 0.0), (0.0, 0.0))
        with pytest.raises(ValueError):
            surface.reaction(particle, state, np.array([0.0, 30.0]))


class TestStaticFrictionCases:
    def test_case_a_rest_without_tangential_force_has_zero_friction(self):
        world, _, _ = make_surface_world()
        result = world.run(2.0)
        np.testing.assert_allclose(result.position, np.zeros_like(result.position), atol=1e-12)
        np.testing.assert_allclose(result.velocity, np.zeros_like(result.velocity), atol=1e-12)

    def test_case_b_small_applied_force_is_balanced_by_static_friction(self):
        world, surface, particle = make_surface_world(ConstantForce((3.0, 0.0)))
        state = State(0.0, (0.0, 0.0), (0.0, 0.0))
        reaction = surface.reaction(particle, state, gravity_force(particle) + np.array([3.0, 0.0]))
        np.testing.assert_allclose(reaction, [-3.0, WEIGHT], atol=1e-12)
        result = world.run(2.0)
        np.testing.assert_allclose(result.position, np.zeros_like(result.position), atol=1e-12)

    def test_case_c_force_at_the_static_threshold_sticks(self):
        # Documented convention: the threshold check is inclusive (<=), so a
        # tangential force exactly equal to mu_s N does not start sliding.
        world, _, _ = make_surface_world(ConstantForce((F_S_MAX, 0.0)))
        result = world.run(1.0)
        np.testing.assert_allclose(result.position, np.zeros_like(result.position), atol=1e-9)

    def test_case_d_force_above_the_threshold_starts_sliding(self):
        applied = 12.0
        acceleration = (applied - F_KINETIC) / MASS
        world, _, _ = make_surface_world(ConstantForce((applied, 0.0)))
        result = world.run(1.5)
        reference = analytical.position(
            result.time, (0.0, 0.0), (0.0, 0.0), (acceleration, 0.0)
        )
        np.testing.assert_allclose(result.position, reference, atol=0.01)
        np.testing.assert_allclose(result.y, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.vy, 0.0, atol=1e-12)

    def test_case_e_sliding_object_decelerates_and_stops_without_jitter(self):
        initial_speed = 5.0
        world, _, _ = make_surface_world(velocity=(initial_speed, 0.0))
        result = world.run(3.0)
        stopped = np.nonzero(result.vx == 0.0)[0]
        assert len(stopped) > 0
        first_stop = int(stopped[0])
        np.testing.assert_array_equal(result.velocity[first_stop:], 0.0)
        position_drift = np.diff(result.position[first_stop:], axis=0)
        np.testing.assert_array_equal(position_drift, 0.0)
        exact_distance = initial_speed**2 / (2.0 * A_KINETIC)
        measured = float(result.x[-1])
        assert measured == pytest.approx(exact_distance, abs=0.01)

    def test_stopping_distance_converges_with_timestep(self):
        exact_distance = 25.0 / (2.0 * A_KINETIC)
        errors = []
        for dt in (0.1, 0.01, 0.001):
            world, _, _ = make_surface_world(velocity=(5.0, 0.0), dt=dt)
            result = world.run(5.0)
            errors.append(abs(float(result.x[-1]) - exact_distance))
        assert errors[0] > errors[1] > errors[2]
        assert errors[2] < 0.01

    def test_frictionless_surface_slides_with_constant_acceleration(self):
        world, _, _ = make_surface_world(ConstantForce((3.0, 0.0)), mu_s=0.0, mu_k=0.0)
        result = world.run(1.0)
        reference = analytical.position(result.time, (0.0, 0.0), (0.0, 0.0), (1.5, 0.0))
        np.testing.assert_allclose(result.position, reference, atol=0.01)


class TestInclineMotion:
    def test_gentle_coefficients_hold_the_block(self):
        angle = math.radians(30.0)
        world, _, _ = make_surface_world(mu_s=0.7, mu_k=0.5, incline_angle=angle)
        result = world.run(2.0)
        np.testing.assert_allclose(result.position, np.zeros_like(result.position), atol=1e-9)

    def test_sliding_incline_matches_exact_constant_acceleration(self):
        angle = math.radians(30.0)
        mu_k = 0.1
        world, surface, _ = make_surface_world(
            mu_s=0.2, mu_k=mu_k, incline_angle=angle, dt=0.001
        )
        result = world.run(1.5)
        acceleration = GRAVITY * (math.sin(angle) - mu_k * math.cos(angle))
        distance = float(np.dot(result.position[-1], surface.tangent))
        exact = -0.5 * acceleration * result.time[-1] ** 2
        assert distance == pytest.approx(exact, abs=0.01)
        perpendicular = result.position @ surface.normal
        np.testing.assert_allclose(perpendicular, 0.0, atol=1e-9)


class TestWorldContactIntegration:
    def test_breakdown_includes_normal_and_friction(self):
        world, _, _ = make_surface_world(ConstantForce((3.0, 0.0)))
        evaluation = world.evaluate_forces(world.initial_state)
        assert [name for name, _ in evaluation.contributions][-2:] == ["normal", "friction"]
        np.testing.assert_allclose(evaluation.net_force, [0.0, 0.0])
        np.testing.assert_allclose(evaluation.acceleration, [0.0, 0.0])

    @pytest.mark.parametrize("direction", [-1.0, 1.0])
    def test_supporting_force_uses_net_deceleration(self, direction):
        world, _, _ = make_surface_world(
            ConstantForce((direction * 4.0, 0.0)), velocity=(direction, 0.0), dt=0.5
        )
        result = world.run_steps(1)
        expected = direction * (1.0 - (F_KINETIC - 4.0) / MASS * 0.5)
        assert result.vx[-1] == pytest.approx(expected)

    def test_zero_friction_does_not_intercept_applied_force_reversal(self):
        world, _, _ = make_surface_world(
            ConstantForce((-4.0, 0.0)), velocity=(1.0, 0.0), mu_s=0.0, mu_k=0.0, dt=1.0
        )
        result = world.run_steps(1)
        assert result.vx[-1] == pytest.approx(-1.0)
        assert result.x[-1] == pytest.approx(1.0)

    def test_world_rejects_non_contact_object(self):
        particle = Particle(MASS, (0.0, 0.0), (0.0, 0.0))
        system = ParticleSystem(particle, [UniformGravity((0.0, -GRAVITY))])
        with pytest.raises(TypeError):
            World(system, dt=0.1, contact=42)

    def test_friction_run_is_deterministic(self):
        first, _, _ = make_surface_world(ConstantForce((12.0, 0.0)))
        second, _, _ = make_surface_world(ConstantForce((12.0, 0.0)))
        left = first.run(1.0)
        right = second.run(1.0)
        np.testing.assert_array_equal(left.position, right.position)
        np.testing.assert_array_equal(left.velocity, right.velocity)

    def test_mixed_deceleration_follows_exact_piecewise_solution(self):
        # Sliding at 5 m/s with a 12 N opposing force: kinetic friction and the
        # applied force stop the block at t_stop, the force exceeds the static
        # threshold, and the block then re-accelerates in the applied direction
        # for the remainder of the step.
        applied = 12.0
        dt = 1.0
        world, _, _ = make_surface_world(
            ConstantForce((-applied, 0.0)), velocity=(5.0, 0.0), dt=dt
        )
        result = world.run_steps(1)
        t_stop = 5.0 / (A_KINETIC + applied / MASS)
        slip_acceleration = (applied - F_KINETIC) / MASS
        expected = -slip_acceleration * (dt - t_stop)
        assert result.vx[-1] == pytest.approx(expected, rel=1e-9)
