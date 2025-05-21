# losses2.py
import torch
import torch.autograd as autograd
import torch.nn.functional as F

def data_loss(model, x, t, u_true):
    u_pred = model(x, t)
    return F.mse_loss(u_pred, u_true)

def pde_loss_wave(model, x_colloc, t_colloc):
    """
    Computes PDE loss for the 1D wave equation: u_tt = c^2 u_xx
    """
    x_colloc.requires_grad_(True)
    t_colloc.requires_grad_(True)

    u = model(x_colloc, t_colloc)

    # First derivatives
    u_t = autograd.grad(u, t_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Second derivatives
    u_tt = autograd.grad(u_t, t_colloc, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x_colloc, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    residual = u_tt - model.c**2 * u_xx

    return torch.mean(residual**2)

# Example usage
if __name__ == "__main__":
    from pinn_model2 import PINN_Wave

    layers = [2, 50, 50, 50, 1]
    model = PINN_Wave(layers)

    x = torch.rand((10, 1), requires_grad=True)
    t = torch.rand((10, 1), requires_grad=True)
    u_true = torch.sin(torch.pi * x) * torch.cos(torch.pi * model.c * t)

    data_l = data_loss(model, x, t, u_true)
    pde_l = pde_loss_wave(model, x, t)

    print(f"Data Loss: {data_l.item()}")
    print(f"PDE Loss : {pde_l.item()}")
