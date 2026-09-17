"""Visualization of recorded simulation data.

This layer performs no physics: it renders only data handed to it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from quasarlab.simulation.world import Trajectory


def plot_trajectory(
    trajectory: Trajectory,
    analytical_positions: NDArray[np.float64] | None = None,
    *,
    ax: Axes | None = None,
    title: str = "Projectile trajectory",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the numerical trajectory and, optionally, an analytical reference.

    ``analytical_positions`` must already be computed by the caller; this
    function does not calculate physics.  When ``ax`` is given, the plot is
    drawn into that existing axes (GUI embedding) instead of a new figure.
    """
    if not isinstance(trajectory, Trajectory):
        raise TypeError(f"trajectory must be a Trajectory, got {type(trajectory).__name__}")

    fig: Figure
    if ax is None:
        fig, target = plt.subplots(figsize=(8, 5))
    else:
        target = ax
        parent = ax.figure
        while not isinstance(parent, Figure):
            parent = parent.figure
        fig = parent
    target.plot(trajectory.x, trajectory.y, "o-", markersize=3, label="Numerical (Euler)")
    if analytical_positions is not None:
        analytical = np.asarray(analytical_positions, dtype=float)
        if analytical.ndim != 2 or analytical.shape[1] != 2:
            raise ValueError(
                f"analytical_positions must have shape (N, 2), got {analytical.shape}"
            )
        target.plot(analytical[:, 0], analytical[:, 1], "k--", label="Analytical")
    target.set_xlabel("x [m]")
    target.set_ylabel("y [m]")
    target.set_title(title)
    target.grid(True, alpha=0.3)
    target.legend()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, target


def plot_component_vs_time(
    trajectory: Trajectory,
    component: str = "y",
    *,
    reference_times: NDArray[np.float64] | None = None,
    reference_values: NDArray[np.float64] | None = None,
    ax: Axes | None = None,
    title: str = "Component versus time",
    ylabel: str = "y [m]",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot one recorded component (x, y, vx, or vy) against time.

    Reference data must already be computed by the caller; this function does
    not calculate physics.  When ``ax`` is given, the plot is drawn into that
    existing axes (GUI embedding) instead of a new figure.
    """
    if not isinstance(trajectory, Trajectory):
        raise TypeError(f"trajectory must be a Trajectory, got {type(trajectory).__name__}")
    allowed = {"x", "y", "vx", "vy"}
    if component not in allowed:
        raise ValueError(f"component must be one of {sorted(allowed)}, got {component!r}")
    values = getattr(trajectory, component)

    fig: Figure
    if ax is None:
        fig, target = plt.subplots(figsize=(8, 5))
    else:
        target = ax
        parent = ax.figure
        while not isinstance(parent, Figure):
            parent = parent.figure
        fig = parent
    target.plot(trajectory.time, values, "o-", markersize=3, label="Numerical (Euler)")
    if reference_times is not None and reference_values is not None:
        times = np.asarray(reference_times, dtype=float)
        reference = np.asarray(reference_values, dtype=float)
        if times.shape != reference.shape or times.ndim != 1:
            raise ValueError("reference_times and reference_values must be 1-D and matching")
        target.plot(times, reference, "k--", label="Analytical")
    target.set_xlabel("t [s]")
    target.set_ylabel(ylabel)
    target.set_title(title)
    target.grid(True, alpha=0.3)
    target.legend()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, target
