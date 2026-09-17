"""Contact mechanics and Coulomb friction on rigid plane surfaces.

This module separates two concepts that must not be conflated:

* **contact mechanics** — surface geometry and the normal force.  The normal
  force is *derived* from the perpendicular force balance
  (``N = -(applied . n)``), never hard-coded to ``m g``; on an incline it
  automatically becomes ``m g cos(theta)`` plus any applied perpendicular
  load.
* **the friction law** — Coulomb friction with static and kinetic branches:
  kinetic friction ``f_k = mu_k N`` opposing the instantaneous sliding
  direction, and static friction supplying exactly the tangential force
  needed to keep the particle at rest, capped at ``f_s,max = mu_s N``.

V0.2 scope and assumptions:

* ideal rigid contact with a *fixed* plane through a point with a unit
  normal; the particle must remain on the surface (free flight and landing
  are not modelled yet and raise ``ValueError``),
* simple Coulomb friction with constant coefficients (``mu_k <= mu_s``),
* no rolling resistance, deformation, or adhesion.

Threshold convention: the sticking check is *inclusive* (``|F_t| <= mu_s N``
sticks), so a tangential force exactly at the static threshold does not
start motion.

Discrete-time stop handling: with explicit Euler, a sliding object whose
kinetic friction would carry it through zero velocity inside one step is
clamped to exactly zero velocity by :meth:`PlaneSurface.constrain` — kinetic
friction can bring the particle to rest, never reverse it.  If the tangential
applied force then exceeds the static threshold, the exact piecewise
solution is used: the particle rests for ``t_stop`` and re-accelerates (in
the applied direction, against kinetic friction) for the remainder of the
step.  The recorded *position* keeps the plain Euler update for that step,
which overestimates the stopping distance by ``O(dt)``; this is a documented,
convergent approximation, not a hidden correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from quasarlab._validation import as_float, as_nonnegative_float, as_vector
from quasarlab.numerical.state import State
from quasarlab.physics.particle import Particle

SURFACE_TOLERANCE = 1e-9


@runtime_checkable
class ContactModel(Protocol):
    """The interface a contact model must satisfy.

    ``reaction`` returns the total contact force (normal + friction) in
    newtons, given the net *applied* (non-contact) force.  ``constrain``
    applies the discrete-time kinematic corrections the contact requires
    after an integrator step (see :meth:`PlaneSurface.constrain`).
    """

    def reaction(
        self, particle: Particle, state: State, applied_force: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def constrain(
        self,
        particle: Particle,
        previous_state: State,
        tentative_state: State,
        applied_force: NDArray[np.float64],
        dt: float,
    ) -> State: ...


@dataclass(frozen=True)
class CoulombFriction:
    """Tangential Coulomb law consuming a supplied normal load in newtons.

    Equality at the static threshold sticks. Nonzero sliding velocity uses
    kinetic friction; no small-speed cutoff is imposed.
    """

    mu_s: float
    mu_k: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu_s", as_nonnegative_float(self.mu_s, "mu_s"))
        object.__setattr__(self, "mu_k", as_nonnegative_float(self.mu_k, "mu_k"))
        if self.mu_k > self.mu_s:
            raise ValueError("mu_k must not exceed mu_s")

    def force(self, velocity: float, applied: float, normal_load: float) -> float:
        """Return signed tangential force in N for signed speed and applied load."""
        velocity = as_float(velocity, "velocity")
        applied = as_float(applied, "applied")
        normal_load = as_nonnegative_float(normal_load, "normal_load")
        if velocity != 0.0:
            return -math.copysign(self.mu_k * normal_load, velocity)
        if abs(applied) <= self.mu_s * normal_load:
            return -applied
        return -math.copysign(self.mu_k * normal_load, applied)


@dataclass
class PlaneSurface:
    """A fixed rigid plane with Coulomb friction.

    ``point`` is any point on the surface and ``normal`` is a nonzero vector
    perpendicular to it pointing toward the allowed side; it is normalized on
    construction.  The unit tangent is ``(n_y, -n_x)`` — for a horizontal
    surface with the normal up, the tangent points along +x; for a surface
    inclined by ``theta`` above horizontal, the tangent points up-slope.
    """

    point: NDArray[np.float64]
    normal: NDArray[np.float64]
    mu_s: float
    mu_k: float

    def __post_init__(self) -> None:
        self.point = as_vector(self.point, "point")
        normal = as_vector(self.normal, "normal")
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError("surface normal must be a nonzero vector")
        self.mu_s = as_nonnegative_float(self.mu_s, "mu_s")
        self.mu_k = as_nonnegative_float(self.mu_k, "mu_k")
        if self.mu_k > self.mu_s:
            raise ValueError(
                f"kinetic friction coefficient ({self.mu_k!r}) must not exceed the"
                f" static friction coefficient ({self.mu_s!r})"
            )
        self.normal = normal / norm
        self.tangent = np.array([self.normal[1], -self.normal[0]])

    def reaction(
        self, particle: Particle, state: State, applied_force: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the total contact force (normal + friction) in newtons.

        ``applied_force`` is the net of all non-contact forces at ``state``.
        The normal magnitude is ``N = -(applied . n)``; a negative value means
        the applied forces pull the particle off the surface, which this model
        cannot represent and raises ``ValueError``.
        """
        contributions = self.force_contributions(particle, state, applied_force)
        return contributions[0][1] + contributions[1][1]

    def force_contributions(
        self, particle: Particle, state: State, applied_force: NDArray[np.float64]
    ) -> tuple[tuple[str, NDArray[np.float64]], ...]:
        """Return separate normal and tangential reactions in newtons."""
        applied = as_vector(applied_force, "applied_force")
        offset = state.position - self.point
        distance = float(np.dot(offset, self.normal))
        tolerance = SURFACE_TOLERANCE * max(1.0, float(np.linalg.norm(offset)))
        if abs(distance) > tolerance:
            raise ValueError(
                "particle is not on the surface; the V0.2 contact model requires"
                " the particle to stay on the surface (free flight and landing"
                " are not supported yet)"
            )
        normal_velocity = float(np.dot(state.velocity, self.normal))
        velocity_tolerance = SURFACE_TOLERANCE * max(1.0, float(np.linalg.norm(state.velocity)))
        if abs(normal_velocity) > velocity_tolerance:
            raise ValueError("velocity must be tangent to the surface; impacts are not supported")
        perpendicular = float(np.dot(applied, self.normal))
        normal_magnitude = -perpendicular
        if normal_magnitude < 0.0:
            raise ValueError(
                "net applied force pulls the particle off the surface (required"
                " normal force is negative); the V0.2 contact model cannot"
                " represent free flight"
            )
        tangential_applied = applied - perpendicular * self.normal
        friction = self._friction(state.velocity, tangential_applied, normal_magnitude)
        return (("normal", normal_magnitude * self.normal), ("friction", friction))

    def _friction(
        self,
        velocity: NDArray[np.float64],
        tangential_applied: NDArray[np.float64],
        normal_magnitude: float,
    ) -> NDArray[np.float64]:
        """Return the friction force for the current contact state."""
        v_t = float(np.dot(velocity, self.tangent))
        f_t = float(np.dot(tangential_applied, self.tangent))
        law = CoulombFriction(self.mu_s, self.mu_k)
        if v_t == 0.0 and abs(f_t) <= self.mu_s * normal_magnitude:
            return -tangential_applied
        return law.force(v_t, f_t, normal_magnitude) * self.tangent

    def constrain(
        self,
        particle: Particle,
        previous_state: State,
        tentative_state: State,
        applied_force: NDArray[np.float64],
        dt: float,
    ) -> State:
        """Apply the discrete-time stop constraint to a tentative Euler step.

        If the particle was sliding and the tentative step carries it through
        zero tangential velocity, kinetic friction would be reversing the
        motion — which the physical law forbids.  The exact piecewise outcome
        under constant forces over the step is applied instead:

        * the particle reaches rest at ``t_stop = |v_t| / (mu_k N/m + opposing
          applied deceleration)`` within the step,
        * if the tangential applied force is within the static threshold it
          then sticks (velocity exactly zero),
        * otherwise it re-accelerates from rest in the applied direction,
          against kinetic friction, for the remainder of the step.

        If the particle was at rest, or does not reach rest within the step,
        the tentative state is returned unchanged.
        """
        applied = as_vector(applied_force, "applied_force")
        normal_magnitude = -float(np.dot(applied, self.normal))
        if normal_magnitude <= 0.0:
            return tentative_state
        v_t_previous = float(np.dot(previous_state.velocity, self.tangent))
        if v_t_previous == 0.0:
            return tentative_state
        v_t_tentative = float(np.dot(tentative_state.velocity, self.tangent))
        if v_t_previous * v_t_tentative >= 0.0:
            return tentative_state

        perpendicular = float(np.dot(applied, self.normal))
        f_t = float(np.dot(applied - perpendicular * self.normal, self.tangent))
        a_friction = self.mu_k * normal_magnitude / particle.mass
        motion_sign = 1.0 if v_t_previous > 0.0 else -1.0
        deceleration = a_friction - motion_sign * f_t / particle.mass
        if deceleration <= 0.0:
            return tentative_state
        t_stop = abs(v_t_previous) / deceleration
        if t_stop > dt:
            return tentative_state

        f_t_magnitude = abs(f_t)
        if f_t_magnitude <= self.mu_s * normal_magnitude:
            stopped_velocity = np.zeros(2, dtype=float)
        else:
            direction = (applied - perpendicular * self.normal) / f_t_magnitude
            slip_acceleration = (f_t_magnitude - self.mu_k * normal_magnitude) / particle.mass
            stopped_velocity = slip_acceleration * (dt - t_stop) * direction
        return State(tentative_state.time, tentative_state.position, stopped_velocity)


def horizontal_surface(mu_s: float, mu_k: float, y: float = 0.0) -> PlaneSurface:
    """Return a horizontal surface at height ``y`` with the normal pointing up."""
    height = as_float(y, "y")
    return PlaneSurface(
        np.array([0.0, height]), np.array([0.0, 1.0]), mu_s, mu_k
    )


def inclined_surface(mu_s: float, mu_k: float, angle: float) -> PlaneSurface:
    """Return a plane through the origin inclined ``angle`` radians above horizontal.

    The surface side facing up is the allowed side; the tangent points
    up-slope (positive x component).
    """
    angle = as_float(angle, "angle")
    return PlaneSurface(
        np.array([0.0, 0.0]),
        np.array([-math.sin(angle), math.cos(angle)]),
        mu_s,
        mu_k,
    )
