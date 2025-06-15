import torch
import torch.nn.functional as F
import torch.autograd as autograd

def data_loss(model, x, t, u_true):
    u_pred = model(x, t)
    return F.mse_loss(u_pred, u_true)

def pde_loss(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
  
    return torch.mean((u_t - model.get_kappa() * u_xx) ** 2)


def initial_condition_loss(model, x0, t0, u0):
    u_pred = model(x0, t0)
    return F.mse_loss(u_pred, u0)

def boundary_condition_loss(model, xb, tb, ub):
    u_pred = model(xb, tb)
    return F.mse_loss(u_pred, ub)
