"""
Training loop and optimization utilities for PhysicsNet-PINN.

The :class:`PINNTrainer` drives the learning process by:
1. Sampling interior collocation, boundary, and initial-condition points.
2. Computing the composite loss (PDE residual + data + BC + IC).
3. Updating model parameters via numerical gradient descent.
4. Recording diagnostics for post-training analysis.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Optional, Dict, List


class PINNTrainer:
    """High-level trainer for a PINN model.

    Parameters
    ----------
    model : PINN
        Neural network model to train.
    equation : object
        Equation instance exposing ``residual``, ``boundary_condition``,
        and ``initial_condition`` methods.
    learning_rate : float
        Initial step size for gradient descent (default ``1e-3``).
    lr_decay : float
        Multiplicative decay applied to the learning rate every
        ``lr_step`` epochs (default ``0.99``).
    lr_step : int
        Epoch interval between learning-rate decays (default ``500``).
    """

    def __init__(
        self,
        model,
        equation,
        learning_rate: float = 1e-3,
        lr_decay: float = 0.99,
        lr_step: int = 500,
    ) -> None:
        self.model = model
        self.equation = equation
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.lr_step = lr_step

        # Diagnostics
        self.loss_history: List[float] = []
        self.lr_history: List[float] = []

    # ---- core training loop -----------------------------------------------

    def train(
        self,
        X_colloc: np.ndarray,
        epochs: int = 5000,
        X_bc: Optional[np.ndarray] = None,
        u_bc: Optional[np.ndarray] = None,
        X_ic: Optional[np.ndarray] = None,
        u_ic: Optional[np.ndarray] = None,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
        verbose: bool = True,
        print_every: int = 500,
        grad_h: float = 1e-6,
    ) -> Dict[str, List[float]]:
        """Train the model using numerical gradient descent.

        Parameters
        ----------
        X_colloc : np.ndarray
            Interior collocation points, shape ``(N, 2)``.
        epochs : int
            Number of training iterations.
        X_bc, u_bc : optional
            Boundary-condition points and target values.
        X_ic, u_ic : optional
            Initial-condition points and target values.
        lambda_pde, lambda_bc, lambda_ic : float
            Loss-term weights.
        verbose : bool
            Print progress to stdout.
        print_every : int
            Epoch interval for console output.
        grad_h : float
            Perturbation size for numerical parameter gradients.

        Returns
        -------
        dict
            ``{"loss": [...], "lr": [...]}``.
        """
        current_lr = self.lr

        for epoch in range(1, epochs + 1):
            # Compute current total loss
            loss = self.model.total_loss(
                X_colloc,
                self.equation,
                X_bc=X_bc,
                u_bc=u_bc,
                X_ic=X_ic,
                u_ic=u_ic,
                lambda_pde=lambda_pde,
                lambda_bc=lambda_bc,
                lambda_ic=lambda_ic,
            )
            self.loss_history.append(loss)
            self.lr_history.append(current_lr)

            # Numerical gradient descent on every parameter array
            for param in self.model.params:
                grad = np.zeros_like(param)
                it = np.nditer(param, flags=["multi_index"])
                while not it.finished:
                    idx = it.multi_index
                    orig = param[idx]
                    param[idx] = orig + grad_h
                    loss_fwd = self.model.total_loss(
                        X_colloc, self.equation,
                        X_bc=X_bc, u_bc=u_bc,
                        X_ic=X_ic, u_ic=u_ic,
                        lambda_pde=lambda_pde,
                        lambda_bc=lambda_bc,
                        lambda_ic=lambda_ic,
                    )
                    param[idx] = orig - grad_h
                    loss_bwd = self.model.total_loss(
                        X_colloc, self.equation,
                        X_bc=X_bc, u_bc=u_bc,
                        X_ic=X_ic, u_ic=u_ic,
                        lambda_pde=lambda_pde,
                        lambda_bc=lambda_bc,
                        lambda_ic=lambda_ic,
                    )
                    param[idx] = orig
                    grad[idx] = (loss_fwd - loss_bwd) / (2.0 * grad_h)
                    it.iternext()

                # Gradient descent step
                param -= current_lr * grad

            # Learning-rate schedule
            if epoch % self.lr_step == 0:
                current_lr *= self.lr_decay

            if verbose and epoch % print_every == 0:
                print(
                    f"Epoch {epoch:>6d}/{epochs}  |  "
                    f"Loss: {loss:.6e}  |  LR: {current_lr:.2e}"
                )

        return {"loss": self.loss_history, "lr": self.lr_history}

    # ---- evaluation -------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        u_exact: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate the trained model.

        Parameters
        ----------
        X_test : np.ndarray
            Test points.
        u_exact : np.ndarray, optional
            Ground-truth values for error computation.

        Returns
        -------
        dict
            Dictionary with ``'mse'`` and (optionally) ``'rel_l2'`` keys.
        """
        u_pred = self.model.forward(X_test)
        metrics: Dict[str, float] = {}
        if u_exact is not None:
            mse = float(np.mean((u_pred - u_exact) ** 2))
            norm = float(np.linalg.norm(u_exact))
            rel_l2 = (
                float(np.linalg.norm(u_pred - u_exact)) / norm
                if norm > 1e-12
                else float("inf")
            )
            metrics["mse"] = mse
            metrics["rel_l2"] = rel_l2
        return metrics

    # ---- persistence ------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save model weights to a ``.npz`` file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``'model.npz'``).
        """
        save_dict = {}
        for i, (w, b) in enumerate(
            zip(self.model.weights, self.model.biases)
        ):
            save_dict[f"w{i}"] = w
            save_dict[f"b{i}"] = b
        np.savez(path, **save_dict)

    def load_model(self, path: str) -> None:
        """Load model weights from a ``.npz`` file.

        Parameters
        ----------
        path : str
            Source file path.
        """
        data = np.load(path)
        for i in range(len(self.model.weights)):
            self.model.weights[i] = data[f"w{i}"]
            self.model.biases[i] = data[f"b{i}"]
