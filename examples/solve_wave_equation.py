"""
Example: Solving the 1-D Wave Equation with PhysicsNet-PINN
============================================================

Solves  ∂²u/∂t² = c² · ∂²u/∂x²  on [0, 1] × [0, 2]
with IC  u(x, 0) = sin(πx), ∂u/∂t(x, 0) = 0
and  BC  u(0, t) = u(1, t) = 0.

Analytical solution:  u(x, t) = sin(πx) · cos(cπt).
"""

import numpy as np
from physicsnet.models import PINN
from physicsnet.equations import WaveEquation
from physicsnet.trainer import PINNTrainer
from physicsnet.utils import (
    generate_collocation_points,
    generate_mesh,
    plot_solution,
    plot_loss_history,
)


def main() -> None:
    # 1. PDE definition
    c = 1.0
    equation = WaveEquation(c=c)

    # 2. Model
    model = PINN(layer_sizes=[2, 32, 32, 1], activation="tanh", seed=7)

    # 3. Collocation points
    X_colloc = generate_collocation_points(500, t_range=(0, 2), seed=1)

    # Boundary conditions
    t_bc = np.linspace(0, 2, 60).reshape(-1, 1)
    X_bc = np.vstack([
        np.hstack([np.zeros_like(t_bc), t_bc]),
        np.hstack([np.ones_like(t_bc), t_bc]),
    ])
    u_bc = equation.boundary_condition(X_bc)

    # Initial condition
    x_ic = np.linspace(0, 1, 60).reshape(-1, 1)
    X_ic = np.hstack([x_ic, np.zeros_like(x_ic)])
    u_ic = equation.initial_condition(X_ic)

    # 4. Train
    trainer = PINNTrainer(model, equation, learning_rate=1e-3)
    trainer.train(
        X_colloc, epochs=2000,
        X_bc=X_bc, u_bc=u_bc,
        X_ic=X_ic, u_ic=u_ic,
        print_every=500,
    )

    # 5. Evaluate
    X_mesh, X_grid, T_grid = generate_mesh(nx=80, nt=80, t_range=(0, 2))
    u_pred = model.forward(X_mesh).reshape(X_grid.shape)
    u_exact = equation.analytical_solution(X_mesh).reshape(X_grid.shape)

    metrics = trainer.evaluate(X_mesh, equation.analytical_solution(X_mesh))
    print(f"\nRelative L2 error: {metrics['rel_l2']:.4%}")

    # 6. Visualize
    plot_solution(X_grid, T_grid, u_pred, title="PINN Prediction — Wave Equation")
    plot_solution(X_grid, T_grid, u_exact, title="Analytical Solution — Wave Equation")
    plot_loss_history(trainer.loss_history, title="Wave Equation — Training Loss")


if __name__ == "__main__":
    main()
