import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# ----- Synthetic Data Generation -----
def generate_synthetic_data(kappa=1.0, N=100, noise_std=0.0):
    Pi = np.pi
    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, N)
    X, T = np.meshgrid(x, t)
    U = np.sin(Pi * X) * np.exp(- (Pi**2) * kappa * T)
    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T.flatten().reshape(-1, 1)
    u_flat = U.flatten().reshape(-1, 1)
    if noise_std > 0:
        u_flat += np.random.normal(0, noise_std, u_flat.shape)
    return x_flat, t_flat, u_flat

# ----- PINN Model -----
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        layer_list = []
        for i in range(len(layers) - 2):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            layer_list.append(nn.Tanh())
        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        self.model = nn.Sequential(*layer_list)
        self.kappa_raw = nn.Parameter(torch.tensor(-3.0))  # Trainable in log-space

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.model(inputs)

    def compute_pde_residual(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = self.forward(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        kappa = torch.exp(self.kappa_raw)
        residual = u_t - kappa * u_xx
        return residual

    def get_kappa(self):
        return torch.exp(self.kappa_raw)

# ----- Loss functions -----
def initial_condition_loss(model, x0, t0, u0):
    u_pred = model(x0, t0)
    return nn.functional.mse_loss(u_pred, u0)

def boundary_condition_loss(model, xb, tb, ub):
    u_pred = model(xb, tb)
    return nn.functional.mse_loss(u_pred, ub)

# ----- Main Execution Block -----
if __name__ == "__main__":
    # ----- Hyperparameters -----
    N_data = 100    # Fewer data points
    N_phys = 10000    # Fewer collocation points
    noise_std = 0.01
    layers = [2, 50, 50, 50, 1]
    true_kappa = 1.0
    num_epochs = 20000   # Fewer epochs for faster training
    lr = 1e-3


    # ----- Load data -----
    x_data_np, t_data_np, u_data_np = generate_synthetic_data(true_kappa, N_data, noise_std)
    x_data = torch.tensor(x_data_np, dtype=torch.float32)
    t_data = torch.tensor(t_data_np, dtype=torch.float32)
    u_data = torch.tensor(u_data_np, dtype=torch.float32)

    # ----- Initial condition (t=0) -----
    x0 = x_data[t_data[:, 0] == 0]
    t0 = t_data[t_data[:, 0] == 0]
    u0 = u_data[t_data[:, 0] == 0]

    # ----- Boundary condition (x=0 and x=1) -----
    xb0 = x_data[x_data[:, 0] == 0]
    tb0 = t_data[x_data[:, 0] == 0]
    ub0 = u_data[x_data[:, 0] == 0]

    xb1 = x_data[x_data[:, 0] == 1]
    tb1 = t_data[x_data[:, 0] == 1]
    ub1 = u_data[x_data[:, 0] == 1]

    xb = torch.cat([xb0, xb1], dim=0)
    tb = torch.cat([tb0, tb1], dim=0)
    ub = torch.cat([ub0, ub1], dim=0)

    # ----- Collocation points -----
    x_phys = torch.rand((N_phys, 1), dtype=torch.float32)
    t_phys = torch.rand((N_phys, 1), dtype=torch.float32)

    # ----- Initialize model -----
    model = PINN(layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # ----- Training -----
    print("Training PINN with IC and BC losses for 20000 epochs...")
    start_time = time.time()
    data_losses = []
    phys_losses = []
    ic_losses = []
    bc_losses = []
    kappa_history = []

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

        data_losses.append(loss_data.item())
        phys_losses.append(loss_phys.item())
        ic_losses.append(loss_ic.item())
        bc_losses.append(loss_bc.item())
        kappa_history.append(model.get_kappa().item())

        if epoch % 1000 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch} | Total Loss: {loss.item():.6f} | Data: {loss_data.item():.6f} | PDE: {loss_phys.item():.6f} | IC: {loss_ic.item():.6f} | BC: {loss_bc.item():.6f} | kappa: {model.get_kappa().item():.6f}")

    end_time = time.time()
    print(f"\nTotal training time: {(end_time - start_time)/60:.2f} minutes")

    # ----- Save model -----
    torch.save(model.state_dict(), "pinn_model_with_ic_bc.pt")

    # ----- Save training logs for comparison -----
    with open("training_logs_with_ic_bc.pkl", "wb") as f:
        pickle.dump({
            "data_losses": data_losses,
            "pde_losses": phys_losses,
            "ic_losses": ic_losses,
            "bc_losses": bc_losses,
            "kappa_values": kappa_history
        }, f)

    # ----- Final trained kappa -----
    final_kappa = model.get_kappa().item()
    print(f"\nFinal trained kappa: {final_kappa:.6f} (True kappa: {true_kappa})")
