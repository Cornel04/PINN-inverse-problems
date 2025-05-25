import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        # Trainable parameter κ
        self.kappa = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# Generate collocation points
def generate_collocation_points(N_f=10000):
    x_f = torch.rand((N_f, 1), device=device)
    t_f = torch.rand((N_f, 1), device=device)
    return x_f.requires_grad_(True), t_f.requires_grad_(True)

# Initial condition: u(x, 0) = sin(pi x)
def generate_initial_condition(N0=100):
    x0 = torch.linspace(0, 1, N0).reshape(-1, 1).to(device)
    t0 = torch.zeros_like(x0).to(device)
    u0 = torch.sin(np.pi * x0).to(device)
    return x0, t0, u0

# Boundary condition: u(0,t) = u(1,t) = 0
def generate_boundary_condition(Nb=100):
    tb = torch.linspace(0, 1, Nb).reshape(-1, 1).to(device)
    x0 = torch.zeros_like(tb).to(device)
    x1 = torch.ones_like(tb).to(device)
    u_b = torch.zeros_like(tb).to(device)
    return x0, x1, tb, u_b

# PDE residual
def pde_residual(model, x, t):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return u_t - model.kappa * u_xx

# Loss function
def loss_function(model, x_f, t_f, x0, t0, u0, xb0, xb1, tb, ub):
    # PDE loss
    f = pde_residual(model, x_f, t_f)
    loss_pde = torch.mean(f**2)

    # Initial condition loss
    u0_pred = model(x0, t0)
    loss_ic = torch.mean((u0_pred - u0)**2)

    # Boundary condition loss
    u_b0 = model(xb0, tb)
    u_b1 = model(xb1, tb)
    loss_bc = torch.mean(u_b0**2 + u_b1**2)

    return loss_pde + loss_ic + loss_bc

# Training
def train(model, epochs=10000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_f, t_f = generate_collocation_points()
    x0, t0, u0 = generate_initial_condition()
    xb0, xb1, tb, ub = generate_boundary_condition()

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model, x_f, t_f, x0, t0, u0, xb0, xb1, tb, ub)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.5f}, κ: {model.kappa.item():.5f}')

    print(f'Final κ estimate: {model.kappa.item():.5f}')


    

# Run
if __name__ == "__main__":
    torch.manual_seed(0)
    model = PINN().to(device)
    train(model, epochs=10000)
