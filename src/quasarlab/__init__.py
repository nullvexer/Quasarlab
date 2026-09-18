"""QuasarLab: a deterministic computational physics laboratory."""

from quasarlab.numerical.integrators import euler_step
from quasarlab.numerical.state import State
from quasarlab.physics.analytical import (
    linear_drag_position,
    linear_drag_velocity,
    quadratic_drag_fall_distance,
    quadratic_drag_fall_speed,
)
from quasarlab.physics.analytical import (
    position as analytical_position,
)
from quasarlab.physics.analytical import (
    velocity as analytical_velocity,
)
from quasarlab.physics.contact import (
    ContactModel,
    CoulombFriction,
    PlaneSurface,
    horizontal_surface,
    inclined_surface,
)
from quasarlab.physics.forces import (
    ConstantForce,
    ForceLaw,
    LinearDrag,
    QuadraticDrag,
    UniformGravity,
    gravitational_force,
)
from quasarlab.physics.particle import Particle
from quasarlab.physics.systems import ParticleSystem
from quasarlab.simulation.world import Trajectory, World

__version__ = "0.2.1"

__all__ = [
    "ConstantForce",
    "ContactModel",
    "CoulombFriction",
    "ForceLaw",
    "LinearDrag",
    "Particle",
    "ParticleSystem",
    "PlaneSurface",
    "QuadraticDrag",
    "State",
    "Trajectory",
    "UniformGravity",
    "World",
    "analytical_position",
    "analytical_velocity",
    "euler_step",
    "gravitational_force",
    "horizontal_surface",
    "inclined_surface",
    "linear_drag_position",
    "linear_drag_velocity",
    "quadratic_drag_fall_distance",
    "quadratic_drag_fall_speed",
]
