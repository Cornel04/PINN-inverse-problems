# losses.py
import torch
import torch.autograd as autograd
import torch.nn.functional as F

def data_loss(model, x, t, u_true):
    """
    Computes the data loss (MSE) between predicted u and true u.
    """
    u_pred = model(x, t)
    return F.mse_loss(u_pred, u_true)


def pde_loss(model, x_colloc, t_colloc):
    """
    Computes PDE residual loss using autograd:
    u_t - kappa * u_xx
    """
    # Ensure gradients are enabled
    x_colloc.requires_grad_(True)
    t_colloc.requires_grad_(True)

    # Compute u from the model
    u = model(x_colloc, t_colloc)

    # Compute gradients u_t and u_x
    u_t = autograd.grad(u, t_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Compute second derivative u_xx
    u_xx = autograd.grad(u_x, x_colloc, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # PDE residual
    residual = u_t - model.kappa * u_xx

    # Return mean squared PDE residual loss
    return torch.mean(residual**2)


# Example usage
if __name__ == "__main__":
    from pinn_model import PINN

    # Example architecture
    layers = [2, 50, 50, 50, 1]
    model = PINN(layers)

    #  data for testing
    x_data = torch.rand((10, 1), requires_grad=True)
    t_data = torch.rand((10, 1), requires_grad=True)
    u_true = torch.sin(torch.pi * x_data) * torch.exp(-torch.pi**2 * model.kappa * t_data)

    x_colloc = torch.rand((10, 1), requires_grad=True)
    t_colloc = torch.rand((10, 1), requires_grad=True)

    # Compute losses
    data_l = data_loss(model, x_data, t_data, u_true)
    pde_l = pde_loss(model, x_colloc, t_colloc)

    print(f"Data Loss: {data_l.item()}")
    print(f"PDE Loss: {pde_l.item()}")
