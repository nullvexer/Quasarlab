"""Run the V0.1 projectile experiment and save its trajectory and plot."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate a 2D projectile under uniform gravity")
    parser.add_argument("--speed", type=float, default=20.0, help="launch speed in m/s")
    parser.add_argument("--angle", type=float, default=45.0, help="launch angle in degrees")
    parser.add_argument("--dt", type=float, default=0.01, help="timestep in seconds")
    parser.add_argument("--show", action="store_true", help="open an interactive plot window")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    from quasarlab.physics import Particle, ParticleSystem, UniformGravity, analytical
    from quasarlab.simulation.world import World
    from quasarlab.visualization.plotting import plot_trajectory

    angle_rad = math.radians(args.angle)
    initial_velocity = np.array(
        [args.speed * math.cos(angle_rad), args.speed * math.sin(angle_rad)]
    )
    gravity = UniformGravity(np.array([0.0, -9.81]))
    particle = Particle(1.0, np.array([0.0, 0.0]), initial_velocity)
    system = ParticleSystem(particle, [gravity])
    world = World(system, dt=args.dt)

    analytical_time_of_flight = 2.0 * initial_velocity[1] / 9.81
    n_steps = math.ceil(analytical_time_of_flight / args.dt)
    duration = n_steps * args.dt
    trajectory = world.run(duration)

    reference_positions = analytical.position(
        trajectory.time, particle.position, initial_velocity, gravity.g
    )
    position_errors = np.linalg.norm(trajectory.position - reference_positions, axis=1)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "projectile_trajectory.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "x_m", "y_m", "vx_m_per_s", "vy_m_per_s"])
        writer.writerows(
            zip(trajectory.time, trajectory.x, trajectory.y, trajectory.vx, trajectory.vy)
        )

    plot_path = output_dir / "projectile.png"
    fig, ax = plot_trajectory(
        trajectory,
        analytical_positions=reference_positions,
        save_path=plot_path,
        show=args.show,
    )
    if not args.show:
        import matplotlib.pyplot as plt

        plt.close(fig)

    print(f"Initial velocity: ({initial_velocity[0]:.6f}, {initial_velocity[1]:.6f}) m/s")
    print(f"Timestep: {args.dt:.6g} s")
    print(f"Simulated duration: {duration:.6f} s")
    print(f"Analytical time of flight: {analytical_time_of_flight:.6f} s")
    print(f"Samples recorded: {len(trajectory)}")
    print(f"Maximum position error vs analytical: {np.max(position_errors):.6g} m")
    print(f"Trajectory CSV: {csv_path}")
    print(f"Trajectory plot: {plot_path}")


if __name__ == "__main__":
    main()
