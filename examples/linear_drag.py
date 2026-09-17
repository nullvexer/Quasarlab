"""Linear drag: falling with F_d = -b v, validated against the exact solution."""

import argparse

import matplotlib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Vertical fall with linear drag")
    parser.add_argument("--b", type=float, default=0.4, help="drag coefficient [kg/s]")
    parser.add_argument("--mass", type=float, default=2.0, help="mass [kg]")
    parser.add_argument("--height", type=float, default=20.0, help="drop height [m]")
    parser.add_argument("--duration", type=float, default=3.0, help="duration [s]")
    parser.add_argument("--dt", type=float, default=0.001, help="timestep [s]")
    parser.add_argument("--show", action="store_true", help="open an interactive plot window")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    from quasarlab import (
        LinearDrag,
        Particle,
        ParticleSystem,
        UniformGravity,
        World,
        linear_drag_position,
        linear_drag_velocity,
    )
    from quasarlab.visualization.plotting import plot_component_vs_time

    if args.b <= 0.0:
        raise SystemExit("b must be positive for this example")
    particle = Particle(args.mass, (0.0, args.height), (0.0, 0.0))
    system = ParticleSystem(particle, [UniformGravity((0.0, -9.81)), LinearDrag(args.b)])
    trajectory = World(system, dt=args.dt).run(args.duration)

    gamma = args.b / args.mass
    v_terminal = -9.81 / gamma  # signed: downward, from F/b with F = -m g
    times = trajectory.time
    exact_vy = linear_drag_velocity(times, 0.0, gamma, v_terminal)
    exact_y = linear_drag_position(times, args.height, 0.0, gamma, v_terminal)
    v_error = float(np.max(np.abs(trajectory.vy - exact_vy)))
    y_error = float(np.max(np.abs(trajectory.y - exact_y)))

    print(f"Terminal velocity (exact, -F/b): {v_terminal:.4f} m/s")
    print(f"Final velocity: {float(trajectory.vy[-1]):.4f} m/s")
    print(f"Max velocity error vs analytical: {v_error:.4g} m/s")
    print(f"Max position error vs analytical: {y_error:.4g} m")

    fig, _ax = plot_component_vs_time(
        trajectory,
        "y",
        reference_times=times,
        reference_values=exact_y,
        title=f"Fall with linear drag (b = {args.b} kg/s)",
    )
    if not args.show:
        import matplotlib.pyplot as plt

        plt.close(fig)


if __name__ == "__main__":
    main()
