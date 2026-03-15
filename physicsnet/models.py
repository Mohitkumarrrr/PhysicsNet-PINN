"""
Neural network architectures for Physics-Informed Neural Networks.

This module implements a fully-connected feedforward neural network and a
residual-block variant, both using pure NumPy.  Numerical differentiation
of the network output with respect to its inputs is used to compute PDE
residuals without requiring an automatic-differentiation framework.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Callable


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation."""
    return np.tanh(z)


def _tanh_deriv(z: np.ndarray) -> np.ndarray:
    """Derivative of tanh."""
    return 1.0 - np.tanh(z) ** 2


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid."""
    s = _sigmoid(z)
    return s * (1.0 - s)


def _relu(z: np.ndarray) -> np.ndarray:
    """Rectified linear unit activation."""
    return np.maximum(0.0, z)


def _relu_deriv(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (z > 0).astype(z.dtype)


_ACTIVATIONS = {
    "tanh": (_tanh, _tanh_deriv),
    "sigmoid": (_sigmoid, _sigmoid_deriv),
    "relu": (_relu, _relu_deriv),
}


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock:
    """A two-layer residual block: out = activation(W2 · activation(W1 · x + b1) + b2) + x.

    The input and output dimensions must match so the skip connection works.

    Parameters
    ----------
    dim : int
        Width of the block (must equal the layer input dimension).
    activation : str
        Name of the activation function (``'tanh'``, ``'sigmoid'``, ``'relu'``).
    """

    def __init__(self, dim: int, activation: str = "tanh") -> None:
        self.dim = dim
        self.act_fn, self.act_deriv = _ACTIVATIONS[activation]

        # Xavier initialization for both sub-layers
        scale = np.sqrt(2.0 / (dim + dim))
        self.W1 = np.random.randn(dim, dim) * scale
        self.b1 = np.zeros((1, dim))
        self.W2 = np.random.randn(dim, dim) * scale
        self.b2 = np.zeros((1, dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the residual block.

        Parameters
        ----------
        x : np.ndarray, shape ``(N, dim)``

        Returns
        -------
        np.ndarray, shape ``(N, dim)``
        """
        z1 = x @ self.W1 + self.b1
        a1 = self.act_fn(z1)
        z2 = a1 @ self.W2 + self.b2
        return self.act_fn(z2 + x)  # skip connection

    @property
    def params(self) -> List[np.ndarray]:
        """Return a flat list of parameter arrays for this block."""
        return [self.W1, self.b1, self.W2, self.b2]


# ---------------------------------------------------------------------------
# PINN Model
# ---------------------------------------------------------------------------

class PINN:
    """Physics-Informed Neural Network implemented in pure NumPy.

    Parameters
    ----------
    layer_sizes : List[int]
        Sequence of layer widths, e.g. ``[2, 64, 64, 64, 1]``.
        The first element is the input dimension and the last is the output
        dimension.
    activation : str, optional
        Activation function name (default ``'tanh'``).
    use_residual : bool, optional
        If ``True``, hidden layers with matching widths are wrapped in
        :class:`ResidualBlock` pairs (default ``False``).
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "tanh",
        use_residual: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.act_fn, self.act_deriv = _ACTIVATIONS[activation]
        self.use_residual = use_residual

        # Build plain dense layers ------------------------------------------
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros((1, fan_out)))

        # Optionally build residual blocks for same-width hidden layers ------
        self.res_blocks: List[Optional[ResidualBlock]] = []
        if use_residual:
            for i in range(1, len(layer_sizes) - 2):
                if layer_sizes[i] == layer_sizes[i + 1]:
                    self.res_blocks.append(
                        ResidualBlock(layer_sizes[i], activation)
                    )
                else:
                    self.res_blocks.append(None)

    # ---- forward pass -----------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the network output for inputs *X*.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, n_inputs)``
            Input collocation points (e.g. columns for *x* and *t*).

        Returns
        -------
        np.ndarray, shape ``(N, n_outputs)``
        """
        a = X
        n_layers = len(self.weights)
        res_idx = 0
        for i in range(n_layers):
            z = a @ self.weights[i] + self.biases[i]
            if i < n_layers - 1:
                a = self.act_fn(z)
                # Apply residual block if available
                if (
                    self.use_residual
                    and res_idx < len(self.res_blocks)
                    and self.res_blocks[res_idx] is not None
                ):
                    a = self.res_blocks[res_idx].forward(a)
                    res_idx += 1
            else:
                a = z  # linear output layer
        return a

    # ---- numerical derivatives --------------------------------------------

    def _numerical_grad(
        self, X: np.ndarray, col: int, h: float = 1e-5
    ) -> np.ndarray:
        """Central-difference approximation of ∂u/∂x_col.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, d)``
        col : int
            Column index to differentiate with respect to.
        h : float
            Finite-difference step size.

        Returns
        -------
        np.ndarray, shape ``(N, 1)``
        """
        X_fwd = X.copy()
        X_bwd = X.copy()
        X_fwd[:, col] += h
        X_bwd[:, col] -= h
        return (self.forward(X_fwd) - self.forward(X_bwd)) / (2.0 * h)

    def _numerical_grad2(
        self, X: np.ndarray, col: int, h: float = 1e-5
    ) -> np.ndarray:
        """Central-difference approximation of ∂²u/∂x_col².

        Parameters
        ----------
        X : np.ndarray, shape ``(N, d)``
        col : int
        h : float

        Returns
        -------
        np.ndarray, shape ``(N, 1)``
        """
        X_fwd = X.copy()
        X_bwd = X.copy()
        X_fwd[:, col] += h
        X_bwd[:, col] -= h
        return (
            self.forward(X_fwd) - 2.0 * self.forward(X) + self.forward(X_bwd)
        ) / (h ** 2)

    # ---- loss components --------------------------------------------------

    def compute_physics_loss(
        self,
        X: np.ndarray,
        equation,
    ) -> float:
        """Mean-squared PDE residual evaluated at collocation points.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, 2)``
            Collocation points (columns: *x*, *t*).
        equation : object
            An equation instance with a ``residual(model, X)`` method.

        Returns
        -------
        float
        """
        residual = equation.residual(self, X)
        return float(np.mean(residual ** 2))

    def compute_data_loss(
        self, X: np.ndarray, u_true: np.ndarray
    ) -> float:
        """Mean-squared error between predictions and reference data.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, d)``
        u_true : np.ndarray, shape ``(N, 1)``

        Returns
        -------
        float
        """
        u_pred = self.forward(X)
        return float(np.mean((u_pred - u_true) ** 2))

    def total_loss(
        self,
        X_colloc: np.ndarray,
        equation,
        X_data: Optional[np.ndarray] = None,
        u_data: Optional[np.ndarray] = None,
        X_bc: Optional[np.ndarray] = None,
        u_bc: Optional[np.ndarray] = None,
        X_ic: Optional[np.ndarray] = None,
        u_ic: Optional[np.ndarray] = None,
        lambda_pde: float = 1.0,
        lambda_data: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
    ) -> float:
        """Weighted sum of physics, data, boundary, and initial-condition losses.

        Parameters
        ----------
        X_colloc : np.ndarray
            Interior collocation points for the PDE residual.
        equation : object
            Equation instance.
        X_data, u_data : optional
            Observation data.
        X_bc, u_bc : optional
            Boundary-condition points and values.
        X_ic, u_ic : optional
            Initial-condition points and values.
        lambda_pde, lambda_data, lambda_bc, lambda_ic : float
            Weighting coefficients for each loss term.

        Returns
        -------
        float
        """
        loss = lambda_pde * self.compute_physics_loss(X_colloc, equation)

        if X_data is not None and u_data is not None:
            loss += lambda_data * self.compute_data_loss(X_data, u_data)
        if X_bc is not None and u_bc is not None:
            loss += lambda_bc * self.compute_data_loss(X_bc, u_bc)
        if X_ic is not None and u_ic is not None:
            loss += lambda_ic * self.compute_data_loss(X_ic, u_ic)

        return loss

    # ---- parameter access -------------------------------------------------

    @property
    def params(self) -> List[np.ndarray]:
        """Flat list of all trainable parameter arrays."""
        p: List[np.ndarray] = []
        for w, b in zip(self.weights, self.biases):
            p.extend([w, b])
        for blk in self.res_blocks:
            if blk is not None:
                p.extend(blk.params)
        return p
