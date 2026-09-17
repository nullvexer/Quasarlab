"""Constant applied force plus gravity: compare Euler with the exact solution."""

import argparse
import math

import matplotlib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate motion under gravity and a constant applied force"
    )
    parser.add_argument("--fx", type=float, default=4.0, help="applied force x [N]")
    parser.add_argument("--fy", type=float, default=0.0, help="applied force y [N]")
    parser.add_argument("--mass", type=float, default=2.0, help="mass [kg]")
    parser.add_argument("--speed", type=float, default=10.0, help="launch speed [m/s]")
    parser.add_argument("--angle", type=float, default=30.0, help="launch angle [deg]")
    parser.add_argument("--dt", type=float, default=0.01, help="timestep [s]")
    parser.add_argument("--show", action="store_true", help="open an interactive plot window")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    from quasarlab import (
        ConstantForce,
        Particle,
        ParticleSystem,
        State,
        UniformGravity,
        World,
        analytical_position,
    )
    from quasarlab.visualization.plotting import plot_trajectory

    angle = math.radians(args.angle)
    initial_velocity = (args.speed * math.cos(angle), args.speed * math.sin(angle))
    particle = Particle(args.mass, (0.0, 0.0), initial_velocity)
    gravity = UniformGravity((0.0, -9.81))
    system = ParticleSystem(particle, [gravity, ConstantForce((args.fx, args.fy))])
    world = World(system, dt=args.dt)
    duration = 2.0 * initial_velocity[1] / 9.81
    trajectory = world.run(math.ceil(duration / args.dt) * args.dt)

    state = State(0.0, particle.position, particle.velocity)
    acceleration = system.acceleration(state)
    reference = analytical_position(
        trajectory.time, particle.position, initial_velocity, acceleration
    )
    error = float(np.max(np.linalg.norm(trajectory.position - reference, axis=1)))

    print(f"Net force (engine):  {system.net_force(state)} N")
    print(f"Acceleration (engine): {acceleration} m/s^2")
    print(f"Max position error vs analytical: {error:.4g} m")

    fig, _ax = plot_trajectory(
        trajectory, analytical_positions=reference, title="Applied force + gravity"
    )
    if not args.show:
        import matplotlib.pyplot as plt

        plt.close(fig)


if __name__ == "__main__":
    main()
