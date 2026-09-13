"""QuasarLab projectile lab: a Tkinter interface on top of the physics engine.

The GUI never computes physics results itself.  It translates form inputs
into structured parameters, hands them to the deterministic engine, and
displays the engine's recorded trajectory.
"""

from __future__ import annotations

import argparse
import math
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from quasarlab import (
    Particle,
    ParticleSystem,
    Trajectory,
    UniformGravity,
    World,
    analytical_position,
)

MAX_STEPS = 200_000
DEFAULT_DURATION_S = 3.0
SELFTEST_MILLISECONDS = 1500


class ProjectileLab:
    """Main application window."""

    def __init__(self, root: tk.Tk, *, interactive: bool = True) -> None:
        self.root = root
        self.interactive = interactive
        self.trajectory: Trajectory | None = None
        self.initial_velocity: tuple[float, float] | None = None
        self.animation: FuncAnimation | None = None
        root.title("QuasarLab - Projectile Lab")
        root.minsize(780, 500)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        self._build_controls()
        self._build_plot()

    def _build_controls(self) -> None:
        panel = ttk.Frame(self.root, padding=10)
        panel.grid(row=0, column=0, sticky="ns")
        self.speed = self._entry(panel, 0, "Launch speed [m/s]", "20.0")
        self.angle = self._entry(panel, 1, "Launch angle [deg]", "45")
        self.mass = self._entry(panel, 2, "Mass [kg]", "1.0")
        self.gravity = self._entry(panel, 3, "Gravity magnitude [m/s^2]", "9.81")
        self.dt = self._entry(panel, 4, "Timestep dt [s]", "0.01")
        ttk.Button(panel, text="Run simulation", command=self.run_simulation).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(12, 4)
        )
        ttk.Button(panel, text="Animate flight", command=self.animate_flight).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=4
        )
        self.info = tk.StringVar(value="Set parameters, then press 'Run simulation'.")
        ttk.Label(panel, textvariable=self.info, wraplength=230, justify="left").grid(
            row=7, column=0, columnspan=2, sticky="nw", pady=(12, 0)
        )
        panel.columnconfigure(1, weight=1)

    def _entry(self, panel: ttk.Frame, row: int, label: str, default: str) -> tk.StringVar:
        ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=2)
        variable = tk.StringVar(value=default)
        ttk.Entry(panel, textvariable=variable, width=12).grid(
            row=row, column=1, sticky="e", pady=2
        )
        return variable

    def _build_plot(self) -> None:
        frame = ttk.Frame(self.root)
        frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.figure = Figure(figsize=(6.0, 4.5), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_xlabel("x [m]")
        self.axis.set_ylabel("y [m]")
        self.axis.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, frame, pack_toolbar=False)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def run_simulation(self) -> None:
        """Read the form, run the deterministic engine, and display the result."""
        try:
            speed = float(self.speed.get())
            angle_deg = float(self.angle.get())
            mass = float(self.mass.get())
            gravity = float(self.gravity.get())
            dt = float(self.dt.get())
            if speed <= 0.0:
                raise ValueError("launch speed must be positive")
            if gravity <= 0.0:
                raise ValueError("gravity magnitude must be positive")
            if dt <= 0.0:
                raise ValueError("timestep dt must be positive")
        except ValueError as exc:
            if not self.interactive:
                raise
            messagebox.showerror("Invalid input", str(exc))
            return

        angle = math.radians(angle_deg)
        initial_velocity = (speed * math.cos(angle), speed * math.sin(angle))
        try:
            particle = Particle(mass, (0.0, 0.0), initial_velocity)
            system = ParticleSystem(particle, [UniformGravity((0.0, -gravity))])
            if initial_velocity[1] > 0.0:
                flight_time = 2.0 * initial_velocity[1] / gravity
            else:
                flight_time = DEFAULT_DURATION_S
            n_steps = math.ceil(flight_time / dt)
            if n_steps > MAX_STEPS:
                raise ValueError(f"dt too small: more than {MAX_STEPS} steps required")
            trajectory = World(system, dt=dt).run(n_steps * dt)
        except (TypeError, ValueError) as exc:
            if not self.interactive:
                raise
            messagebox.showerror("Simulation error", str(exc))
            return

        self._stop_animation()
        self.trajectory = trajectory
        self.initial_velocity = initial_velocity
        self._plot_trajectory(trajectory, initial_velocity, gravity)
        self._report(trajectory, initial_velocity, gravity)

    def _plot_trajectory(
        self,
        trajectory: Trajectory,
        initial_velocity: tuple[float, float],
        gravity: float,
    ) -> None:
        self.axis.clear()
        self.axis.plot(trajectory.x, trajectory.y, "o-", markersize=3, label="Numerical (Euler)")
        reference = analytical_position(
            trajectory.time, (0.0, 0.0), initial_velocity, (0.0, -gravity)
        )
        self.axis.plot(reference[:, 0], reference[:, 1], "k--", label="Analytical")
        self.axis.set_xlabel("x [m]")
        self.axis.set_ylabel("y [m]")
        self.axis.set_title("Projectile trajectory")
        self.axis.grid(True, alpha=0.3)
        self.axis.legend()
        self.canvas.draw_idle()

    def _report(
        self,
        trajectory: Trajectory,
        initial_velocity: tuple[float, float],
        gravity: float,
    ) -> None:
        reference = analytical_position(
            trajectory.time, (0.0, 0.0), initial_velocity, (0.0, -gravity)
        )
        max_error = float(np.max(np.abs(trajectory.position - reference)))
        self.info.set(
            f"Samples: {len(trajectory)}\n"
            f"Max height: {float(np.max(trajectory.y)):.2f} m\n"
            f"End position x: {float(trajectory.x[-1]):.2f} m\n"
            f"Simulated time: {float(trajectory.time[-1]):.2f} s\n"
            f"Max error vs analytical: {max_error:.4f} m"
        )

    def animate_flight(self) -> None:
        """Replay the recorded trajectory as a moving point (no new physics)."""
        if self.trajectory is None:
            if not self.interactive:
                raise RuntimeError("no simulation result to animate")
            messagebox.showinfo("No result", "Run a simulation first.")
            return
        self._stop_animation()
        trajectory = self.trajectory
        (point,) = self.axis.plot([], [], "ro", markersize=6)
        if len(trajectory.time) > 1:
            step_ms = float(trajectory.time[1] - trajectory.time[0])
        else:
            step_ms = 0.01
        interval_ms = max(1, int(round(1000.0 * step_ms)))

        def update(frame: int) -> None:
            point.set_data([trajectory.x[frame]], [trajectory.y[frame]])

        self.animation = FuncAnimation(
            self.figure,
            update,
            frames=len(trajectory.time),
            interval=interval_ms,
            blit=False,
            repeat=False,
        )
        self.canvas.draw_idle()

    def _stop_animation(self) -> None:
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuasarLab projectile lab (graphical)")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run one default simulation, close automatically, exit 0 on success",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    lab = ProjectileLab(root, interactive=not args.selftest)
    if args.selftest:
        lab.run_simulation()
        root.after(SELFTEST_MILLISECONDS, root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
