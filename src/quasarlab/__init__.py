"""QuasarLab: a deterministic computational physics laboratory."""

from quasarlab.numerical.integrators import euler_step
from quasarlab.numerical.state import State
from quasarlab.physics.analytical import position as analytical_position
from quasarlab.physics.analytical import velocity as analytical_velocity
from quasarlab.physics.forces import UniformGravity, gravitational_force
from quasarlab.physics.particle import Particle
from quasarlab.physics.systems import ParticleSystem
from quasarlab.simulation.world import Trajectory, World

__version__ = "0.1.0"

__all__ = [
    "Particle",
    "ParticleSystem",
    "State",
    "Trajectory",
    "UniformGravity",
    "World",
    "analytical_position",
    "analytical_velocity",
    "euler_step",
    "gravitational_force",
]
