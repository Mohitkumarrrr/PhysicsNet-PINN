"""
Unit tests for PhysicsNet-PINN core components.

Run with:
    python -m pytest tests/test_models.py -v
or:
    python -m unittest tests.test_models
"""

import unittest
import numpy as np

from physicsnet.models import PINN, ResidualBlock
from physicsnet.equations import HeatEquation, WaveEquation, BurgersEquation
from physicsnet.trainer import PINNTrainer
from physicsnet.utils import generate_collocation_points, generate_mesh


class TestPINNModel(unittest.TestCase):
    """Tests for the PINN neural network model."""

    def setUp(self) -> None:
        self.model = PINN(layer_sizes=[2, 16, 16, 1], activation="tanh", seed=0)
        self.X = np.random.rand(50, 2)

    def test_forward_output_shape(self) -> None:
        """Forward pass returns the correct output shape."""
        out = self.model.forward(self.X)
        self.assertEqual(out.shape, (50, 1))

    def test_forward_deterministic(self) -> None:
        """Repeated forward passes produce identical results."""
        out1 = self.model.forward(self.X)
        out2 = self.model.forward(self.X)
        np.testing.assert_array_equal(out1, out2)

    def test_numerical_grad_shape(self) -> None:
        """Numerical gradient has the right shape."""
        grad = self.model._numerical_grad(self.X, col=0)
        self.assertEqual(grad.shape, (50, 1))

    def test_numerical_grad2_shape(self) -> None:
        """Second numerical derivative has the right shape."""
        grad2 = self.model._numerical_grad2(self.X, col=1)
        self.assertEqual(grad2.shape, (50, 1))


class TestResidualBlock(unittest.TestCase):
    """Tests for the ResidualBlock."""

    def test_forward_shape(self) -> None:
        block = ResidualBlock(dim=16, activation="tanh")
        x = np.random.rand(30, 16)
        out = block.forward(x)
        self.assertEqual(out.shape, (30, 16))

    def test_params_count(self) -> None:
        block = ResidualBlock(dim=8)
        # W1, b1, W2, b2 → 4 arrays
        self.assertEqual(len(block.params), 4)


class TestEquations(unittest.TestCase):
    """Tests for PDE equation classes."""

    def setUp(self) -> None:
        self.model = PINN(layer_sizes=[2, 8, 1], seed=1)
        self.X = np.random.rand(20, 2)

    def test_heat_residual_shape(self) -> None:
        eq = HeatEquation(alpha=0.01)
        r = eq.residual(self.model, self.X)
        self.assertEqual(r.shape, (20, 1))

    def test_wave_residual_shape(self) -> None:
        eq = WaveEquation(c=1.0)
        r = eq.residual(self.model, self.X)
        self.assertEqual(r.shape, (20, 1))

    def test_burgers_residual_shape(self) -> None:
        eq = BurgersEquation()
        r = eq.residual(self.model, self.X)
        self.assertEqual(r.shape, (20, 1))

    def test_heat_analytical(self) -> None:
        """Analytical solution at t = 0 should equal sin(πx)."""
        eq = HeatEquation(alpha=0.05)
        x = np.linspace(0, 1, 50).reshape(-1, 1)
        X = np.hstack([x, np.zeros_like(x)])
        u = eq.analytical_solution(X)
        np.testing.assert_allclose(u, np.sin(np.pi * x), atol=1e-12)


class TestTrainer(unittest.TestCase):
    """Basic trainer smoke tests."""

    def test_evaluate_returns_metrics(self) -> None:
        model = PINN(layer_sizes=[2, 8, 1], seed=2)
        eq = HeatEquation()
        trainer = PINNTrainer(model, eq)
        X = np.random.rand(30, 2)
        u_exact = eq.analytical_solution(X)
        metrics = trainer.evaluate(X, u_exact)
        self.assertIn("mse", metrics)
        self.assertIn("rel_l2", metrics)
        self.assertGreaterEqual(metrics["mse"], 0.0)


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""

    def test_collocation_shape(self) -> None:
        pts = generate_collocation_points(100, seed=0)
        self.assertEqual(pts.shape, (100, 2))

    def test_mesh_shape(self) -> None:
        X_mesh, X_grid, T_grid = generate_mesh(nx=10, nt=20)
        self.assertEqual(X_mesh.shape, (200, 2))
        self.assertEqual(X_grid.shape, (20, 10))
        self.assertEqual(T_grid.shape, (20, 10))


if __name__ == "__main__":
    unittest.main()
