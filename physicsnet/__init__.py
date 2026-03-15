"""
PhysicsNet-PINN — Physics-Informed Neural Networks for Solving PDEs
====================================================================

A lightweight, pure-NumPy library for solving partial differential equations
using physics-informed neural networks. Supports the heat equation, wave
equation, and Burgers' equation out of the box, with an extensible API for
custom PDEs.

Example
-------
>>> from physicsnet.models import PINN
>>> from physicsnet.equations import HeatEquation
>>> from physicsnet.trainer import PINNTrainer
>>> model = PINN(layer_sizes=[2, 64, 64, 1])
>>> equation = HeatEquation(alpha=0.01)
>>> trainer = PINNTrainer(model, equation, learning_rate=1e-3)
"""

__version__ = "1.0.0"
__author__ = "Mohit Kumar"

from physicsnet.models import PINN, ResidualBlock
from physicsnet.equations import HeatEquation, WaveEquation, BurgersEquation
from physicsnet.trainer import PINNTrainer
from physicsnet.utils import (
    generate_collocation_points,
    generate_mesh,
    plot_solution,
    plot_loss_history,
)

__all__ = [
    "PINN",
    "ResidualBlock",
    "HeatEquation",
    "WaveEquation",
    "BurgersEquation",
    "PINNTrainer",
    "generate_collocation_points",
    "generate_mesh",
    "plot_solution",
    "plot_loss_history",
]
