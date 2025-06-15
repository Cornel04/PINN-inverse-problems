import numpy as np
import matplotlib.pyplot as plt
import torch
from pinn_bc_ic import PINN, generate_synthetic_data, initial_condition_loss, boundary_condition_loss
import torch.nn as nn
import torch.optim as optim
import time

def train_pinn(noise_std, true_kappa=1.0, N_data=100, N_phys=1000, num_epochs=20000, lr=1e-3, layers=[2, 50, 50, 50, 1]):
    # Generate data
    x_data_np, t_data_np, u_data_np = generate_synthetic_data(true_kappa, N_data, noise_std)
    x_data = torch.tensor(x_data_np, dtype=torch.float32)
    t_data = torch.tensor(t_data_np, dtype=torch.float32)
    u_data = torch.tensor(u_data_np, dtype=torch.float32)

    # Initial condition
    x0 = x_data[t_data[:, 0] == 0]
    t0 = t_data[t_data[:, 0] == 0]
    u0 = u_data[t_data[:, 0] == 0]

    # Boundary condition (x=0 and x=1)
    xb0 = x_data[x_data[:, 0] == 0]
    tb0 = t_data[x_data[:, 0] == 0]
    ub0 = u_data[x_data[:, 0] == 0]
    xb1 = x_data[x_data[:, 0] == 1]
    tb1 = t_data[x_data[:, 0] == 1]
    ub1 = u_data[x_data[:, 0] == 1]
    xb = torch.cat([xb0, xb1], dim=0)
    tb = torch.cat([tb0, tb1], dim=0)
    ub = torch.cat([ub0, ub1], dim=0)

    # Collocation points
    x_phys = torch.rand((N_phys, 1), dtype=torch.float32)
    t_phys = torch.rand((N_phys, 1), dtype=torch.float32)

    # Initialize model
    model = PINN(layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Training
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        u_pred = model(x_data, t_data)
        loss_data = mse(u_pred, u_data)

        residual = model.compute_pde_residual(x_phys, t_phys)
        loss_phys = mse(residual, torch.zeros_like(residual))

        loss_ic = initial_condition_loss(model, x0, t0, u0)
        loss_bc = boundary_condition_loss(model, xb, tb, ub)

        loss = loss_data + loss_phys + loss_ic + loss_bc
        loss.backward()
        optimizer.step()

    return model.get_kappa().item()

if __name__ == "__main__":
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    estimated_kappas = []

    for noise in noise_levels:
        print(f"\nTraining for noise level: {noise}")
        kappa_est = train_pinn(noise_std=noise)
        estimated_kappas.append(kappa_est)
        print(f"Estimated kappa: {kappa_est:.6f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, estimated_kappas, 'o-', label="Estimated kappa")
    plt.axhline(y=1.0, color='r', linestyle='--', label="True kappa")
    plt.xlabel("Noise standard deviation")
    plt.ylabel("Estimated kappa")
    plt.title("Noise Robustness of PINN")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pinn_noise_robustness.png", dpi=300)
    plt.show()
