"""QuasarLab physics lab: a Tkinter interface on top of the physics engine.

The GUI never computes physics results itself.  Scenarios translate form
inputs into structured engine parameters (initial conditions, force laws,
contact model, duration); the deterministic engine performs every physical
computation; the GUI displays recorded engine data, engine force breakdowns,
and references computed by the engine's own analytical functions.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import io
import math
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import numpy as np
import sv_ttk
from matplotlib import style as mpl_style
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from quasarlab import (
    ConstantForce,
    LinearDrag,
    Particle,
    ParticleSystem,
    QuadraticDrag,
    State,
    Trajectory,
    UniformGravity,
    World,
    analytical_position,
    horizontal_surface,
    inclined_surface,
    linear_drag_position,
    quadratic_drag_fall_distance,
)
from quasarlab.physics.contact import ContactModel
from quasarlab.visualization.plotting import plot_component_vs_time, plot_trajectory

MAX_STEPS = 200_000
SELFTEST_MILLISECONDS = 1500
LIVE_UPDATE_DELAY_MS = 300
DEFAULT_DURATION_S = 3.0
MPL_DARK_STYLE = "seaborn-v0_8-darkgrid"
MPL_LIGHT_STYLE = "seaborn-v0_8-whitegrid"


def _enable_dpi_awareness() -> None:
    """Request DPI awareness so Windows renders the UI sharply on scaled displays.

    Best effort only: falls back to the legacy user32 call and never raises,
    so behavior on other platforms is unchanged.
    """
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass


def _application_icon_png() -> bytes:
    """Render the QuasarLab application icon (a stylized Q) as PNG bytes."""
    figure = Figure(figsize=(1, 1), dpi=64)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.add_patch(
        Circle((0.5, 0.58), 0.28, fill=False, edgecolor="#4C9AFF", linewidth=7.0)
    )
    axis.plot(
        [0.60, 0.82],
        [0.30, 0.10],
        color="#4C9AFF",
        linewidth=7.0,
        solid_capstyle="round",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", transparent=True)
    return buffer.getvalue()


@dataclass(frozen=True)
class FieldSpec:
    """One GUI parameter: a key, a label with units, and a default value."""

    key: str
    label: str
    default: str


@dataclass(frozen=True)
class SimulationSetup:
    """Everything the engine needs for one scenario run.

    All quantities come from engine objects; ``extra_info`` carries
    engine-derived summary values (e.g. a terminal speed).
    """

    system: ParticleSystem
    contact: ContactModel | None
    duration: float
    plot_title: str
    extra_info: str = ""


class Scenario:
    """A named experiment: parameter fields plus engine assembly and display."""

    name = ""
    fields: tuple[FieldSpec, ...] = ()

    def build(self, values: dict[str, float]) -> SimulationSetup:
        """Translate form values into engine objects (no physics results here)."""
        raise NotImplementedError

    def plot(self, axis, trajectory: Trajectory, setup: SimulationSetup, values: dict[str, float]):
        """Render the recorded data; return the (x, y) path used for animation."""
        raise NotImplementedError

    def report(
        self, trajectory: Trajectory, setup: SimulationSetup, values: dict[str, float]
    ) -> str:
        """Return scenario-specific lines built from engine-computed data."""
        return ""


def _flight_duration(initial_velocity_y: float, gravity: float, dt: float) -> float:
    """Duration covering the ballistic arc (experiment design, not physics)."""
    if initial_velocity_y > 0.0:
        flight_time = 2.0 * initial_velocity_y / gravity
    else:
        flight_time = DEFAULT_DURATION_S
    return math.ceil(flight_time / dt) * dt


class ProjectileScenario(Scenario):
    name = "Projectile (gravity)"
    fields = (
        FieldSpec("speed", "Launch speed [m/s]", "20.0"),
        FieldSpec("angle", "Launch angle [deg]", "45"),
        FieldSpec("gravity", "Gravity [m/s^2]", "9.81"),
        FieldSpec("mass", "Mass [kg]", "1.0"),
        FieldSpec("dt", "Timestep dt [s]", "0.01"),
    )

    def build(self, values):
        speed = values["speed"]
        angle = math.radians(values["angle"])
        gravity = values["gravity"]
        initial_velocity = (speed * math.cos(angle), speed * math.sin(angle))
        particle = Particle(values["mass"], (0.0, 0.0), initial_velocity)
        system = ParticleSystem(particle, [UniformGravity((0.0, -gravity))])
        return SimulationSetup(
            system=system,
            contact=None,
            duration=_flight_duration(initial_velocity[1], gravity, values["dt"]),
            plot_title="Projectile trajectory",
        )

    def plot(self, axis, trajectory, setup, values):
        reference = analytical_position(
            trajectory.time,
            (0.0, 0.0),
            setup.system.particle.velocity,
            (0.0, -values["gravity"]),
        )
        plot_trajectory(
            trajectory, analytical_positions=reference, ax=axis, title=setup.plot_title
        )
        return trajectory.x, trajectory.y

    def report(self, trajectory, setup, values):
        reference = analytical_position(
            trajectory.time,
            (0.0, 0.0),
            setup.system.particle.velocity,
            (0.0, -values["gravity"]),
        )
        max_error = float(np.max(np.abs(trajectory.position - reference)))
        return (
            f"Max height: {float(np.max(trajectory.y)):.2f} m\n"
            f"Max error vs analytical: {max_error:.4f} m"
        )


class AppliedForceScenario(Scenario):
    name = "Applied force"
    fields = (
        FieldSpec("speed", "Launch speed [m/s]", "10.0"),
        FieldSpec("angle", "Launch angle [deg]", "30"),
        FieldSpec("fx", "Applied force Fx [N]", "4.0"),
        FieldSpec("fy", "Applied force Fy [N]", "0.0"),
        FieldSpec("gravity", "Gravity [m/s^2]", "9.81"),
        FieldSpec("mass", "Mass [kg]", "2.0"),
        FieldSpec("dt", "Timestep dt [s]", "0.01"),
    )

    def build(self, values):
        speed = values["speed"]
        angle = math.radians(values["angle"])
        gravity = values["gravity"]
        initial_velocity = (speed * math.cos(angle), speed * math.sin(angle))
        particle = Particle(values["mass"], (0.0, 0.0), initial_velocity)
        system = ParticleSystem(
            particle,
            [UniformGravity((0.0, -gravity)), ConstantForce((values["fx"], values["fy"]))],
        )
        return SimulationSetup(
            system=system,
            contact=None,
            duration=_flight_duration(initial_velocity[1], gravity, values["dt"]),
            plot_title="Motion under gravity and a constant applied force",
        )

    def plot(self, axis, trajectory, setup, values):
        # Constant net force -> constant acceleration; the acceleration is the
        # engine's own Newton-second-law result at the initial state.
        acceleration = setup.system.acceleration(
            State(0.0, setup.system.particle.position, setup.system.particle.velocity)
        )
        reference = analytical_position(
            trajectory.time,
            (0.0, 0.0),
            setup.system.particle.velocity,
            acceleration,
        )
        plot_trajectory(
            trajectory, analytical_positions=reference, ax=axis, title=setup.plot_title
        )
        return trajectory.x, trajectory.y

    def report(self, trajectory, setup, values):
        acceleration = setup.system.acceleration(
            State(0.0, setup.system.particle.position, setup.system.particle.velocity)
        )
        reference = analytical_position(
            trajectory.time, (0.0, 0.0), setup.system.particle.velocity, acceleration
        )
        max_error = float(np.max(np.abs(trajectory.position - reference)))
        return f"Max error vs analytical: {max_error:.4f} m"


class LinearDragScenario(Scenario):
    name = "Linear drag (falling)"
    fields = (
        FieldSpec("b", "Drag coefficient b [kg/s]", "0.4"),
        FieldSpec("gravity", "Gravity [m/s^2]", "9.81"),
        FieldSpec("mass", "Mass [kg]", "2.0"),
        FieldSpec("height", "Drop height [m]", "20.0"),
        FieldSpec("duration", "Duration [s]", "3.0"),
        FieldSpec("dt", "Timestep dt [s]", "0.001"),
    )

    def build(self, values):
        particle = Particle(values["mass"], (0.0, values["height"]), (0.0, 0.0))
        system = ParticleSystem(
            particle,
            [UniformGravity((0.0, -values["gravity"])), LinearDrag(values["b"])],
        )
        return SimulationSetup(
            system=system,
            contact=None,
            duration=values["duration"],
            plot_title="Vertical fall with linear drag",
        )

    def plot(self, axis, trajectory, setup, values):
        gamma = values["b"] / values["mass"]
        v_terminal = -values["mass"] * values["gravity"] / values["b"]
        if values["b"] > 0.0:
            reference = linear_drag_position(
                trajectory.time, values["height"], 0.0, gamma, v_terminal
            )
        else:
            reference = None
        plot_component_vs_time(
            trajectory,
            "y",
            reference_times=None if reference is None else trajectory.time,
            reference_values=reference,
            ax=axis,
            title=setup.plot_title,
        )
        return trajectory.time, trajectory.y

    def report(self, trajectory, setup, values):
        if values["b"] <= 0.0:
            return ""
        v_terminal = -values["mass"] * values["gravity"] / values["b"]
        return (
            f"Terminal velocity: {v_terminal:.3f} m/s (exact: F/b)\n"
            f"Final velocity: {float(trajectory.vy[-1]):.3f} m/s"
        )


class QuadraticDragScenario(Scenario):
    name = "Quadratic drag (falling)"
    fields = (
        FieldSpec("density", "Fluid density [kg/m^3]", "1.225"),
        FieldSpec("cd", "Drag coefficient Cd", "1.0"),
        FieldSpec("area", "Reference area [m^2]", "0.5"),
        FieldSpec("gravity", "Gravity [m/s^2]", "9.81"),
        FieldSpec("mass", "Mass [kg]", "80.0"),
        FieldSpec("height", "Drop height [m]", "500.0"),
        FieldSpec("duration", "Duration [s]", "15.0"),
        FieldSpec("dt", "Timestep dt [s]", "0.001"),
    )

    def build(self, values):
        law = QuadraticDrag(values["density"], values["cd"], values["area"])
        particle = Particle(values["mass"], (0.0, values["height"]), (0.0, 0.0))
        system = ParticleSystem(
            particle, [UniformGravity((0.0, -values["gravity"])), law]
        )
        extra = ""
        if law.drag_factor > 0.0:
            terminal = law.terminal_speed(values["mass"], values["gravity"])
            extra = f"Terminal speed (engine): {terminal:.3f} m/s\n"
        return SimulationSetup(
            system=system,
            contact=None,
            duration=values["duration"],
            plot_title="Vertical fall with quadratic drag",
            extra_info=extra,
        )

    def plot(self, axis, trajectory, setup, values):
        law = QuadraticDrag(values["density"], values["cd"], values["area"])
        reference = None
        if law.drag_factor > 0.0:
            terminal = law.terminal_speed(values["mass"], values["gravity"])
            reference = values["height"] - quadratic_drag_fall_distance(
                trajectory.time, terminal, values["gravity"]
            )
        plot_component_vs_time(
            trajectory,
            "y",
            reference_times=None if reference is None else trajectory.time,
            reference_values=reference,
            ax=axis,
            title=setup.plot_title,
        )
        return trajectory.time, trajectory.y

    def report(self, trajectory, setup, values):
        return f"Final speed: {abs(float(trajectory.vy[-1])):.3f} m/s"


class FrictionScenario(Scenario):
    name = "Friction on a surface"
    fields = (
        FieldSpec("mu_s", "Static friction mu_s", "0.5"),
        FieldSpec("mu_k", "Kinetic friction mu_k", "0.3"),
        FieldSpec("fx", "Applied horizontal force [N]", "12.0"),
        FieldSpec("incline", "Incline angle [deg] (0 = flat)", "0.0"),
        FieldSpec("v0", "Initial speed along surface [m/s]", "0.0"),
        FieldSpec("gravity", "Gravity [m/s^2]", "9.81"),
        FieldSpec("mass", "Mass [kg]", "2.0"),
        FieldSpec("duration", "Duration [s]", "3.0"),
        FieldSpec("dt", "Timestep dt [s]", "0.001"),
    )

    def build(self, values):
        angle = math.radians(values["incline"])
        if values["incline"] == 0.0:
            surface = horizontal_surface(values["mu_s"], values["mu_k"])
        else:
            surface = inclined_surface(values["mu_s"], values["mu_k"], angle)
        particle = Particle(
            values["mass"],
            (0.0, 0.0),
            (values["v0"] * math.cos(angle), values["v0"] * math.sin(angle)),
        )
        system = ParticleSystem(
            particle,
            [
                UniformGravity((0.0, -values["gravity"])),
                ConstantForce((values["fx"], 0.0)),
            ],
        )
        return SimulationSetup(
            system=system,
            contact=surface,
            duration=values["duration"],
            plot_title="Block on a surface with Coulomb friction",
        )

    def plot(self, axis, trajectory, setup, values):
        plot_trajectory(trajectory, ax=axis, title=setup.plot_title)
        return trajectory.x, trajectory.y

    def report(self, trajectory, setup, values):
        traveled = float(np.linalg.norm(trajectory.position[-1] - trajectory.position[0]))
        return f"Distance traveled: {traveled:.3f} m"


SCENARIOS: tuple[Scenario, ...] = (
    ProjectileScenario(),
    AppliedForceScenario(),
    LinearDragScenario(),
    QuadraticDragScenario(),
    FrictionScenario(),
)


class PhysicsLab:
    """Main application window: scenarios, live updates, engine results."""

    def __init__(self, root: tk.Tk, *, interactive: bool = True) -> None:
        self.root = root
        self.interactive = interactive
        self.trajectory: Trajectory | None = None
        self.animation: FuncAnimation | None = None
        self._animation_path: tuple[np.ndarray, np.ndarray] | None = None
        self._refresh_job: str | None = None
        self._variables: dict[str, tk.StringVar] = {}
        self._dark_mode = True
        self._icon: tk.PhotoImage | None = None
        root.title("QuasarLab - Physics Lab")
        self._apply_theme()
        self._apply_icon()
        root.minsize(1000, 620)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        self._build_controls()
        self._build_plot()

    def _apply_theme(self) -> None:
        """Apply the ttk theme, base font, and matching matplotlib style."""
        try:
            sv_ttk.set_theme("dark" if self._dark_mode else "light")
        except tk.TclError:
            pass
        mpl_style.use(MPL_DARK_STYLE if self._dark_mode else MPL_LIGHT_STYLE)
        style = ttk.Style()
        default_font = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica", 11)
        style.configure(".", font=default_font)

    def _apply_icon(self) -> None:
        """Set the window icon; a missing icon must never block startup."""
        try:
            self._icon = tk.PhotoImage(data=base64.b64encode(_application_icon_png()))
            self.root.iconphoto(True, self._icon)
        except tk.TclError:
            self._icon = None

    def _toggle_theme(self) -> None:
        """Switch light/dark and restyle both the widgets and the plot."""
        self._dark_mode = not self._dark_mode
        self._stop_animation()
        self._apply_theme()
        self._style_results_text()
        self._rebuild_plot()
        self._compute_and_show(show_errors=False)

    def _style_results_text(self) -> None:
        """Match the read-only results panel colors to the active ttk theme."""
        style = ttk.Style()
        background = style.lookup("TFrame", "background") or "#1c1c1c"
        foreground = style.lookup("TLabel", "foreground") or "#fafafa"
        self._results_text.configure(
            background=background,
            foreground=foreground,
            insertbackground=foreground,
        )

    def _current_scenario(self) -> Scenario:
        for scenario in SCENARIOS:
            if scenario.name == self._scenario_selector.get():
                return scenario
        return SCENARIOS[0]

    def _build_controls(self) -> None:
        panel = ttk.Frame(self.root, padding=16)
        panel.grid(row=0, column=0, sticky="ns")

        scenario_frame = ttk.LabelFrame(panel, text="Scenario", padding=10)
        scenario_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._scenario_selector = tk.StringVar(value=SCENARIOS[0].name)
        selector = ttk.Combobox(
            scenario_frame,
            textvariable=self._scenario_selector,
            values=[scenario.name for scenario in SCENARIOS],
            state="readonly",
            width=28,
        )
        selector.grid(row=0, column=0, sticky="ew")
        selector.bind("<<ComboboxSelected>>", self._on_scenario_changed)
        ttk.Checkbutton(
            scenario_frame,
            text="Light mode",
            command=self._toggle_theme,
            style="Switch.TCheckbutton",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        parameters_frame = ttk.LabelFrame(panel, text="Parameters", padding=10)
        parameters_frame.grid(row=1, column=0, sticky="nsew", pady=6)
        self._parameter_frame = ttk.Frame(parameters_frame)
        self._parameter_frame.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        results_frame = ttk.LabelFrame(panel, text="Results", padding=10)
        results_frame.grid(row=2, column=0, sticky="nsew", pady=(6, 0))
        ttk.Button(results_frame, text="Run simulation", command=self.run_simulation).grid(
            row=0, column=0, sticky="ew", pady=6
        )
        ttk.Button(results_frame, text="Animate motion", command=self.animate_motion).grid(
            row=1, column=0, sticky="ew", pady=6
        )
        self._results_text = tk.Text(
            results_frame,
            height=14,
            width=34,
            wrap="word",
            state="disabled",
            relief="flat",
        )
        scrollbar = ttk.Scrollbar(
            results_frame, orient="vertical", command=self._results_text.yview
        )
        self._results_text.configure(yscrollcommand=scrollbar.set)
        self._results_text.grid(row=2, column=0, sticky="nsew", pady=(6, 0))
        scrollbar.grid(row=2, column=1, sticky="ns", pady=(6, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(2, weight=1)
        self._style_results_text()

        self._set_results("Edit any value - the plot updates automatically.")
        self._rebuild_parameter_panel()

    def _set_results(self, text: str) -> None:
        """Render a report into the read-only results panel."""
        self._results_text.configure(state="normal")
        self._results_text.delete("1.0", "end")
        self._results_text.insert("1.0", text)
        self._results_text.configure(state="disabled")

    def _rebuild_parameter_panel(self) -> None:
        for child in self._parameter_frame.winfo_children():
            child.destroy()
        self._variables = {}
        scenario = self._current_scenario()
        for row, field in enumerate(scenario.fields):
            variable = tk.StringVar(value=field.default)
            ttk.Label(self._parameter_frame, text=field.label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=6
            )
            ttk.Entry(self._parameter_frame, textvariable=variable, width=12).grid(
                row=row, column=1, sticky="e", pady=6
            )
            self._variables[field.key] = variable
            variable.trace_add("write", self._on_parameter_changed)
        self._request_refresh()

    def _build_plot(self) -> None:
        self._plot_frame = ttk.Frame(self.root)
        self._plot_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        mpl_style.use(MPL_DARK_STYLE if self._dark_mode else MPL_LIGHT_STYLE)
        self.figure = Figure(figsize=(7.2, 5.4), dpi=120)
        self.axis = self.figure.add_subplot(111)
        self.axis.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self._plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self._plot_frame, pack_toolbar=False)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _rebuild_plot(self) -> None:
        """Recreate the plot area so a theme change restyles the whole figure."""
        self._plot_frame.destroy()
        self._build_plot()

    def run_simulation(self) -> None:
        """Read the form, run the deterministic engine, and display the result."""
        self._compute_and_show(show_errors=True)

    def _on_scenario_changed(self, *_args: object) -> None:
        self._rebuild_parameter_panel()

    def _on_parameter_changed(self, *_args: object) -> None:
        """Debounce live updates so typing does not trigger a run per keystroke."""
        if self._refresh_job is not None:
            self.root.after_cancel(self._refresh_job)
        self._refresh_job = self.root.after(LIVE_UPDATE_DELAY_MS, self._auto_refresh)

    def _request_refresh(self) -> None:
        self._on_parameter_changed()

    def _auto_refresh(self) -> None:
        self._refresh_job = None
        self._compute_and_show(show_errors=False)

    def _compute_and_show(self, *, show_errors: bool) -> None:
        scenario = self._current_scenario()
        try:
            values = {key: float(variable.get()) for key, variable in self._variables.items()}
            if values["dt"] <= 0.0:
                raise ValueError("timestep dt must be positive")
        except ValueError as exc:
            if not self.interactive:
                raise
            if show_errors:
                messagebox.showerror("Invalid input", str(exc))
            else:
                self._set_results(f"Waiting for valid input: {exc}")
            return

        try:
            setup = scenario.build(values)
            dt = values["dt"]
            n_steps = math.ceil(setup.duration / dt)
            if n_steps > MAX_STEPS:
                raise ValueError(f"dt too small: more than {MAX_STEPS} steps required")
            trajectory = World(setup.system, dt=dt, contact=setup.contact).run(n_steps * dt)
        except (TypeError, ValueError) as exc:
            if not self.interactive:
                raise
            if show_errors:
                messagebox.showerror("Simulation error", str(exc))
            else:
                self._set_results(f"Cannot simulate: {exc}")
            return

        self._stop_animation()
        self.trajectory = trajectory
        # Clear the axes so repeated runs never accumulate plot artists or
        # duplicate legend entries; plotting.py stays reusable by examples.
        self.axis.clear()
        self.axis.grid(True, alpha=0.3)
        self._animation_path = scenario.plot(self.axis, trajectory, setup, values)
        self._set_results(self._build_report(scenario, trajectory, setup, values))

    def _build_report(
        self,
        scenario: Scenario,
        trajectory: Trajectory,
        setup: SimulationSetup,
        values: dict[str, float],
    ) -> str:
        final_state = State(
            float(trajectory.time[-1]),
            trajectory.position[-1],
            trajectory.velocity[-1],
        )
        lines = [
            f"Samples: {len(trajectory)}   Simulated time: {float(trajectory.time[-1]):.2f} s",
            f"Final position: ({float(trajectory.x[-1]):.2f}, {float(trajectory.y[-1]):.2f}) m",
            f"Final velocity: ({float(trajectory.vx[-1]):.2f}, {float(trajectory.vy[-1]):.2f}) m/s",
        ]
        if setup.extra_info:
            lines.append(setup.extra_info.rstrip("\n"))
        scenario_lines = scenario.report(trajectory, setup, values)
        if scenario_lines:
            lines.append(scenario_lines)
        lines.append("Forces at final state (engine):")
        for label, force in setup.system.force_contributions(final_state):
            lines.append(f"  {label}: ({force[0]:.2f}, {force[1]:.2f}) N")
        applied = setup.system.net_force(final_state)
        if setup.contact is not None:
            try:
                reaction = setup.contact.reaction(
                    setup.system.particle, final_state, applied
                )
            except ValueError:
                lines.append("  contact: n/a")
            else:
                lines.append(
                    f"  contact (normal+friction): ({reaction[0]:.2f}, {reaction[1]:.2f}) N"
                )
                applied = applied + reaction
        lines.append(f"  net force: ({applied[0]:.2f}, {applied[1]:.2f}) N")
        acceleration = applied / setup.system.particle.mass
        lines.append(f"  acceleration: ({acceleration[0]:.2f}, {acceleration[1]:.2f}) m/s^2")
        return "\n".join(lines)

    def animate_motion(self) -> None:
        """Replay the recorded trajectory as a moving point (no new physics)."""
        if self.trajectory is None or self._animation_path is None:
            if not self.interactive:
                raise RuntimeError("no simulation result to animate")
            messagebox.showinfo("No result", "Run a simulation first.")
            return
        self._stop_animation()
        path_x, path_y = self._animation_path
        (point,) = self.axis.plot([], [], "ro", markersize=6)
        if len(self.trajectory.time) > 1:
            step_ms = float(self.trajectory.time[1] - self.trajectory.time[0])
        else:
            step_ms = 0.01
        interval_ms = max(1, int(round(1000.0 * step_ms)))

        def update(frame: int) -> None:
            point.set_data([path_x[frame]], [path_y[frame]])

        self.animation = FuncAnimation(
            self.figure,
            update,
            frames=len(self.trajectory.time),
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
    parser = argparse.ArgumentParser(description="QuasarLab physics lab (graphical)")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run one default simulation, close automatically, exit 0 on success",
    )
    return parser.parse_args()


def main() -> int:
    _enable_dpi_awareness()
    args = parse_args()
    root = tk.Tk()
    lab = PhysicsLab(root, interactive=not args.selftest)
    if args.selftest:
        lab.run_simulation()
        root.after(SELFTEST_MILLISECONDS, root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
