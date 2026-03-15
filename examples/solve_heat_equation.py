"""
Example: Solving the 1-D Heat Equation with PhysicsNet-PINN
============================================================

Solves  ∂u/∂t = α · ∂²u/∂x²  on [0, 1] × [0, 1]
with IC  u(x, 0) = sin(πx)  and BC  u(0, t) = u(1, t) = 0.

The analytical solution is  u(x, t) = sin(πx) · exp(−α π² t).
"""

import numpy as np
from physicsnet.models import PINN
from physicsnet.equations import HeatEquation
from physicsnet.trainer import PINNTrainer
from physicsnet.utils import (
    generate_collocation_points,
    generate_mesh,
    plot_solution,
    plot_loss_history,
)


def main() -> None:
    # 1. Define the PDE
    alpha = 0.01
    equation = HeatEquation(alpha=alpha)

    # 2. Build the PINN model (2 inputs → 1 output, three hidden layers)
    model = PINN(layer_sizes=[2, 32, 32, 1], activation="tanh", seed=42)

    # 3. Generate training points
    X_colloc = generate_collocation_points(500, seed=0)

    # Boundary conditions: x = 0 and x = 1 for several time values
    t_bc = np.linspace(0, 1, 50).reshape(-1, 1)
    X_bc = np.vstack([
        np.hstack([np.zeros_like(t_bc), t_bc]),
        np.hstack([np.ones_like(t_bc), t_bc]),
    ])
    u_bc = equation.boundary_condition(X_bc)

    # Initial condition: t = 0 for several x values
    x_ic = np.linspace(0, 1, 50).reshape(-1, 1)
    X_ic = np.hstack([x_ic, np.zeros_like(x_ic)])
    u_ic = equation.initial_condition(X_ic)

    # 4. Train
    trainer = PINNTrainer(model, equation, learning_rate=1e-3)
    trainer.train(
        X_colloc,
        epochs=2000,
        X_bc=X_bc, u_bc=u_bc,
        X_ic=X_ic, u_ic=u_ic,
        print_every=500,
    )

    # 5. Evaluate on a fine grid
    X_mesh, X_grid, T_grid = generate_mesh(nx=80, nt=80)
    u_pred = model.forward(X_mesh).reshape(X_grid.shape)
    u_exact = equation.analytical_solution(X_mesh).reshape(X_grid.shape)

    metrics = trainer.evaluate(X_mesh, equation.analytical_solution(X_mesh))
    print(f"\nRelative L2 error: {metrics['rel_l2']:.4%}")
    print(f"MSE:               {metrics['mse']:.6e}")

    # 6. Plot results
    plot_solution(X_grid, T_grid, u_pred, title="PINN Prediction — Heat Equation")
    plot_solution(X_grid, T_grid, u_exact, title="Analytical Solution — Heat Equation")
    plot_solution(
        X_grid, T_grid, np.abs(u_pred - u_exact),
        title="Absolute Error", cmap="hot",
    )
    plot_loss_history(trainer.loss_history, title="Heat Equation — Training Loss")


if __name__ == "__main__":
    main()
