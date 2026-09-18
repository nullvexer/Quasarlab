"""V0.2.1 engine hardening: immutability, validation, and contact abstraction.

These tests describe the V0.2.1 contract before the engine is refactored:

* force laws and surfaces cannot be mutated into invalid physics,
* frozen result objects hand out write-protected arrays,
* ForceEvaluation maintains F_net = sum(contributions) by construction,
* World depends only on the ContactModel protocol (no PlaneSurface special
  case) and reports contact contributions through a generic evaluation,
* validation rejects booleans and blank labels everywhere.
"""

import dataclasses

import numpy as np
import pytest

from quasarlab import (
    ConstantForce,
    CoulombFriction,
    LinearDrag,
    Particle,
    QuadraticDrag,
    UniformGravity,
    horizontal_surface,
)


def make_particle(velocity=(0.0, 0.0)) -> Particle:
    return Particle(2.0, (0.0, 0.0), velocity)


class TestForceLawImmutability:
    def test_force_laws_are_frozen_dataclasses(self):
        for law in (UniformGravity(), ConstantForce((1.0, 2.0)), LinearDrag(0.5)):
            assert dataclasses.is_dataclass(law)
            assert law.__dataclass_params__.frozen

    def test_gravity_vector_cannot_be_mutated(self):
        law = UniformGravity((0.0, -9.81))
        with pytest.raises(ValueError):
            law.g[0] = 5.0
        np.testing.assert_allclose(law.g, [0.0, -9.81])

    def test_gravity_attribute_reassignment_rejected(self):
        law = UniformGravity((0.0, -9.81))
        with pytest.raises(dataclasses.FrozenInstanceError):
            law.g = np.array([1.0, 0.0])

    def test_constant_force_vector_cannot_be_mutated(self):
        law = ConstantForce((1.0, 2.0))
        with pytest.raises(ValueError):
            law.force_value[0] = 99.0
        np.testing.assert_allclose(law.force_value, [1.0, 2.0])

    def test_drag_coefficient_reassignment_rejected(self):
        law = LinearDrag(0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            law.b = -1.0

    def test_quadratic_drag_reassignment_rejected(self):
        law = QuadraticDrag(2.0, 1.0, 0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            law.density = -5.0

    def test_plane_surface_geometry_cannot_be_mutated(self):
        surface = horizontal_surface(0.5, 0.3)
        with pytest.raises(dataclasses.FrozenInstanceError):
            surface.normal = np.array([1.0, 0.0])
        np.testing.assert_allclose(surface.normal, [0.0, 1.0])

    def test_coulomb_friction_is_frozen_and_public(self):
        law = CoulombFriction(0.5, 0.3)
        with pytest.raises(dataclasses.FrozenInstanceError):
            law.mu_s = 0.1
        assert law.force(1.0, 0.0, 10.0) == pytest.approx(-3.0)
