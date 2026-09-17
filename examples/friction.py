"""Coulomb friction on a horizontal surface: static threshold and kinetic slide."""

import argparse

import matplotlib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Block on a surface with friction")
    parser.add_argument("--mu-s", type=float, default=0.5, help="static friction coefficient")
    parser.add_argument("--mu-k", type=float, default=0.3, help="kinetic friction coefficient")
    parser.add_argument("--force", type=float, default=12.0, help="applied force [N]")
    parser.add_argument("--mass", type=float, default=2.0, help="mass [kg]")
    parser.add_argument("--duration", type=float, default=2.0, help="duration [s]")
    parser.add_argument("--dt", type=float, default=0.001, help="timestep [s]")
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
        UniformGravity,
        World,
        horizontal_surface,
    )
    from quasarlab.visualization.plotting import plot_component_vs_time

    gravity = 9.81
    weight = args.mass * gravity
    surface = horizontal_surface(args.mu_s, args.mu_k)
    particle = Particle(args.mass, (0.0, 0.0), (0.0, 0.0))
    system = ParticleSystem(
        particle, [UniformGravity((0.0, -gravity)), ConstantForce((args.force, 0.0))]
    )
    world = World(system, dt=args.dt, contact=surface)

    static_limit = args.mu_s * weight
    kinetic = args.mu_k * weight
    state = world.initial_state
    breakdown = world.evaluate_forces(state)
    print("Forces at rest (engine breakdown):")
    for label, force in breakdown.contributions:
        print(f"  {label}: ({force[0]:.3f}, {force[1]:.3f}) N")

    trajectory = world.run(args.duration)
    if args.force > static_limit:
        acceleration = (args.force - kinetic) / args.mass
        reference = np.column_stack(
            [0.5 * acceleration * trajectory.time**2, np.zeros_like(trajectory.time)]
        )
        error = float(np.max(np.linalg.norm(trajectory.position - reference, axis=1)))
        print(f"Sliding: analytical acceleration {acceleration:.3f} m/s^2")
        print(f"Max position error vs analytical: {error:.4g} m")
    else:
        print(
            f"Static: |F| = {abs(args.force):.3f} N <= mu_s N = {static_limit:.3f} N, block holds"
        )

    fig, _ax = plot_component_vs_time(
        trajectory,
        "x",
        ylabel="x [m]",
        title=f"Friction on a surface (F = {args.force:.1f} N)",
    )
    if not args.show:
        import matplotlib.pyplot as plt

        plt.close(fig)


if __name__ == "__main__":
    main()
