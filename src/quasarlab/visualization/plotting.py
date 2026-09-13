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
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the numerical trajectory and, optionally, an analytical reference.

    ``analytical_positions`` must already be computed by the caller; this
    function does not calculate physics.
    """
    if not isinstance(trajectory, Trajectory):
        raise TypeError(f"trajectory must be a Trajectory, got {type(trajectory).__name__}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trajectory.x, trajectory.y, "o-", markersize=3, label="Numerical (Euler)")
    if analytical_positions is not None:
        analytical = np.asarray(analytical_positions, dtype=float)
        if analytical.ndim != 2 or analytical.shape[1] != 2:
            raise ValueError(
                f"analytical_positions must have shape (N, 2), got {analytical.shape}"
            )
        ax.plot(analytical[:, 0], analytical[:, 1], "k--", label="Analytical")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Projectile trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
