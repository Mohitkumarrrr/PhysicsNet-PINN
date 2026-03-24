"""
Utility functions for data generation, mesh creation, and visualization.

These helpers are used by the examples and the trainer to prepare
collocation points and to plot results.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


# ---------------------------------------------------------------------------
# Data / mesh generation
# ---------------------------------------------------------------------------

def generate_collocation_points(
    n_points: int,
    x_range: Tuple[float, float] = (0.0, 1.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample random interior collocation points in the (x, t) domain.

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    x_range : tuple of float
        Spatial domain bounds ``(x_min, x_max)``.
    t_range : tuple of float
        Temporal domain bounds ``(t_min, t_max)``.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray, shape ``(n_points, 2)``
        Columns are ``[x, t]``.
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(x_range[0], x_range[1], (n_points, 1))
    t = np.random.uniform(t_range[0], t_range[1], (n_points, 1))
    return np.hstack([x, t])


def generate_mesh(
    nx: int = 100,
    nt: int = 100,
    x_range: Tuple[float, float] = (0.0, 1.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a regular 2-D grid over the (x, t) domain.

    Parameters
    ----------
    nx, nt : int
        Number of grid points in the spatial and temporal dimensions.
    x_range, t_range : tuple of float
        Domain bounds.

    Returns
    -------
    X_mesh : np.ndarray, shape ``(nx*nt, 2)``
        Flattened grid points.
    X_grid : np.ndarray, shape ``(nt, nx)``
        Meshgrid *x* values (for plotting).
    T_grid : np.ndarray, shape ``(nt, nx)``
        Meshgrid *t* values (for plotting).
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    t = np.linspace(t_range[0], t_range[1], nt)
    X_grid, T_grid = np.meshgrid(x, t)
    X_mesh = np.hstack(
        [X_grid.reshape(-1, 1), T_grid.reshape(-1, 1)]
    )
    return X_mesh, X_grid, T_grid


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_solution(
    X_grid: np.ndarray,
    T_grid: np.ndarray,
    U: np.ndarray,
    title: str = "PINN Solution",
    cmap: str = "viridis",
    save_path: Optional[str] = None,
) -> None:
    """Plot a 2-D heatmap of the solution field u(x, t).

    Parameters
    ----------
    X_grid, T_grid : np.ndarray
        Meshgrid arrays from :func:`generate_mesh`.
    U : np.ndarray
        Solution values reshaped to match the grid.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    save_path : str, optional
        If provided, save the figure to this path instead of displaying it.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(X_grid, T_grid, U, cmap=cmap, shading="auto")
    fig.colorbar(pcm, ax=ax, label="u(x, t)")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_loss_history(
    loss_history: List[float],
    title: str = "Training Loss",
    save_path: Optional[str] = None,
) -> None:
    """Plot the training loss over epochs on a log scale.

    Parameters
    ----------
    loss_history : list of float
        Loss values recorded during training.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure instead of displaying it.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(loss_history, linewidth=1.2, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def normalize_data(data):
    """Normalize input data to [0,1] range."""
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val + 1e-8)
