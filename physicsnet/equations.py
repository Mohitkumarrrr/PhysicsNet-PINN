"""
PDE definitions for Physics-Informed Neural Networks.

Each equation class exposes three methods consumed by the trainer:

* ``residual(model, X)`` — PDE residual evaluated at collocation points.
* ``boundary_condition(X)`` — Dirichlet boundary values.
* ``initial_condition(X)`` — Initial-state values.

All inputs follow the convention ``X[:, 0] = x`` and ``X[:, 1] = t``.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicsnet.models import PINN


# ---------------------------------------------------------------------------
# Heat Equation:  ∂u/∂t = α · ∂²u/∂x²
# ---------------------------------------------------------------------------

class HeatEquation:
    """1-D heat (diffusion) equation with thermal diffusivity *α*.

    Analytical solution (used for validation):
        u(x, t) = sin(π x) · exp(−α π² t)

    Default boundary conditions: u(0, t) = u(1, t) = 0.
    Default initial condition:   u(x, 0) = sin(π x).

    Parameters
    ----------
    alpha : float
        Thermal diffusivity coefficient (default ``0.01``).
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self.name = "heat"

    def residual(self, model: "PINN", X: np.ndarray) -> np.ndarray:
        """Compute r = ∂u/∂t − α · ∂²u/∂x².

        Parameters
        ----------
        model : PINN
        X : np.ndarray, shape ``(N, 2)``

        Returns
        -------
        np.ndarray, shape ``(N, 1)``
        """
        u_t = model._numerical_grad(X, col=1)     # ∂u/∂t
        u_xx = model._numerical_grad2(X, col=0)    # ∂²u/∂x²
        return u_t - self.alpha * u_xx

    def boundary_condition(self, X: np.ndarray) -> np.ndarray:
        """Dirichlet BC: u = 0 at x = 0 and x = 1.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, 2)``

        Returns
        -------
        np.ndarray, shape ``(N, 1)``
        """
        return np.zeros((X.shape[0], 1))

    def initial_condition(self, X: np.ndarray) -> np.ndarray:
        """IC: u(x, 0) = sin(π x).

        Parameters
        ----------
        X : np.ndarray, shape ``(N, 2)``

        Returns
        -------
        np.ndarray, shape ``(N, 1)``
        """
        return np.sin(np.pi * X[:, 0:1])

    def analytical_solution(self, X: np.ndarray) -> np.ndarray:
        """Exact solution u(x, t) = sin(π x) exp(−α π² t).

        Parameters
        ----------
        X : np.ndarray, shape ``(N, 2)``

        Returns
        -------
        np.ndarray, shape ``(N, 1)``
        """
        x = X[:, 0:1]
        t = X[:, 1:2]
        return np.sin(np.pi * x) * np.exp(-self.alpha * np.pi ** 2 * t)


# ---------------------------------------------------------------------------
# Wave Equation:  ∂²u/∂t² = c² · ∂²u/∂x²
# ---------------------------------------------------------------------------

class WaveEquation:
    """1-D wave equation with wave speed *c*.

    Analytical solution:
        u(x, t) = sin(π x) · cos(c π t)

    Default boundary conditions: u(0, t) = u(1, t) = 0.
    Default initial condition:   u(x, 0) = sin(π x), ∂u/∂t(x, 0) = 0.

    Parameters
    ----------
    c : float
        Wave propagation speed (default ``1.0``).
    """

    def __init__(self, c: float = 1.0) -> None:
        self.c = c
        self.name = "wave"

    def residual(self, model: "PINN", X: np.ndarray) -> np.ndarray:
        """Compute r = ∂²u/∂t² − c² · ∂²u/∂x².

        Parameters
        ----------
        model : PINN
        X : np.ndarray, shape ``(N, 2)``

        Returns
        -------
        np.ndarray, shape ``(N, 1)``
        """
        u_tt = model._numerical_grad2(X, col=1)
        u_xx = model._numerical_grad2(X, col=0)
        return u_tt - self.c ** 2 * u_xx

    def boundary_condition(self, X: np.ndarray) -> np.ndarray:
        """Dirichlet BC: u = 0 at x = 0 and x = 1."""
        return np.zeros((X.shape[0], 1))

    def initial_condition(self, X: np.ndarray) -> np.ndarray:
        """IC: u(x, 0) = sin(π x)."""
        return np.sin(np.pi * X[:, 0:1])

    def analytical_solution(self, X: np.ndarray) -> np.ndarray:
        """Exact solution u(x, t) = sin(π x) cos(c π t)."""
        x = X[:, 0:1]
        t = X[:, 1:2]
        return np.sin(np.pi * x) * np.cos(self.c * np.pi * t)


# ---------------------------------------------------------------------------
# Burgers' Equation:  ∂u/∂t + u · ∂u/∂x = ν · ∂²u/∂x²
# ---------------------------------------------------------------------------

class BurgersEquation:
    """1-D viscous Burgers' equation with kinematic viscosity *nu*.

    No closed-form solution is stored; training relies on the PDE residual
    and boundary / initial conditions only.

    Default domain: x ∈ [−1, 1], t ∈ [0, 1].
    Default IC: u(x, 0) = −sin(π x).
    Default BC: u(−1, t) = u(1, t) = 0.

    Parameters
    ----------
    nu : float
        Kinematic viscosity (default ``0.01 / π``).
    """

    def __init__(self, nu: float = 0.01 / np.pi) -> None:
        self.nu = nu
        self.name = "burgers"

    def residual(self, model: "PINN", X: np.ndarray) -> np.ndarray:
        """Compute r = ∂u/∂t + u · ∂u/∂x − ν · ∂²u/∂x².

        Parameters
        ----------
        model : PINN
        X : np.ndarray, shape ``(N, 2)``

        Returns
        -------
        np.ndarray, shape ``(N, 1)``
        """
        u = model.forward(X)
        u_t = model._numerical_grad(X, col=1)
        u_x = model._numerical_grad(X, col=0)
        u_xx = model._numerical_grad2(X, col=0)
        return u_t + u * u_x - self.nu * u_xx

    def boundary_condition(self, X: np.ndarray) -> np.ndarray:
        """Dirichlet BC: u = 0 at x = −1 and x = 1."""
        return np.zeros((X.shape[0], 1))

    def initial_condition(self, X: np.ndarray) -> np.ndarray:
        """IC: u(x, 0) = −sin(π x)."""
        return -np.sin(np.pi * X[:, 0:1])


class PoissonEquation:
    """2D Poisson equation solver."""
    def __init__(self, source_term=None):
        self.source_term = source_term
    def residual(self, x, y, u, u_xx, u_yy):
        f = self.source_term(x, y) if self.source_term else 0
        return u_xx + u_yy - f
