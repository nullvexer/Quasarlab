"""Quadratic drag: terminal speed and the exact tanh fall solution."""

import argparse

import matplotlib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Vertical fall with quadratic drag")
    parser.add_argument("--density", type=float, default=1.225, help="fluid density [kg/m^3]")
    parser.add_argument("--cd", type=float, default=1.0, help="drag coefficient Cd")
    parser.add_argument("--area", type=float, default=0.5, help="reference area [m^2]")
    parser.add_argument("--mass", type=float, default=80.0, help="mass [kg]")
    parser.add_argument("--height", type=float, default=500.0, help="drop height [m]")
    parser.add_argument("--duration", type=float, default=15.0, help="duration [s]")
    parser.add_argument("--dt", type=float, default=0.001, help="timestep [s]")
    parser.add_argument("--show", action="store_true", help="open an interactive plot window")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    from quasarlab import (
        Particle,
        ParticleSystem,
        QuadraticDrag,
        UniformGravity,
        World,
        quadratic_drag_fall_distance,
        quadratic_drag_fall_speed,
    )
    from quasarlab.visualization.plotting import plot_component_vs_time

    law = QuadraticDrag(args.density, args.cd, args.area)
    if law.drag_factor <= 0.0:
        raise SystemExit("this example requires positive drag parameters")
    particle = Particle(args.mass, (0.0, args.height), (0.0, 0.0))
    system = ParticleSystem(particle, [UniformGravity((0.0, -9.81)), law])
    trajectory = World(system, dt=args.dt).run(args.duration)

    v_terminal = law.terminal_speed(args.mass, 9.81)
    times = trajectory.time
    exact_speed = quadratic_drag_fall_speed(times, v_terminal, 9.81)
    exact_distance = quadratic_drag_fall_distance(times, v_terminal, 9.81)
    speed_error = float(np.max(np.abs(-trajectory.vy - exact_speed)))
    distance_error = float(np.max(np.abs((args.height - trajectory.y) - exact_distance)))

    print(f"Terminal speed (engine): sqrt(2 m g / (rho Cd A)) = {v_terminal:.3f} m/s")
    print(f"Final speed: {abs(float(trajectory.vy[-1])):.3f} m/s")
    print(f"Max speed error vs analytical: {speed_error:.4g} m/s")
    print(f"Max fallen-distance error vs analytical: {distance_error:.4g} m")

    fig, _ax = plot_component_vs_time(
        trajectory,
        "y",
        reference_times=times,
        reference_values=args.height - exact_distance,
        title=f"Fall with quadratic drag (v_t = {v_terminal:.1f} m/s)",
    )
    if not args.show:
        import matplotlib.pyplot as plt

        plt.close(fig)


if __name__ == "__main__":
    main()
