# train.py

import torch
import torch.optim as optim
from pinn_model import PINN
from generate_data import generate_synthetic_data
from losses import data_loss, pde_loss

# -----------------------------
# Hyperparameters
# -----------------------------
N = 100                    # Number of points in each dimension
noise_std = 0.01           # Noise level in synthetic data
layers = [2, 50, 50, 50, 1]
pretrain_iters = 20000
finetune_iters = 10000
lr_pretrain = 1e-3

# -----------------------------
# Generate training data
# -----------------------------
x_np, t_np, u_np = generate_synthetic_data(kappa=1.0, N=N, noise_std=noise_std)

x_train = torch.tensor(x_np, dtype=torch.float32)
t_train = torch.tensor(t_np, dtype=torch.float32)
u_train = torch.tensor(u_np, dtype=torch.float32)

# Use same points as collocation points
x_colloc = x_train.clone().detach().requires_grad_(True)
t_colloc = t_train.clone().detach().requires_grad_(True)

# -----------------------------
# Initialize model
# -----------------------------
model = PINN(layers)

# -----------------------------
# Pre-training: minimize data loss only
# -----------------------------
optimizer_pre = optim.Adam(model.parameters(), lr=lr_pretrain)

print("\n Pre-training (data loss only)...")
for it in range(pretrain_iters):
    optimizer_pre.zero_grad()
    loss_d = data_loss(model, x_train, t_train, u_train)
    loss_d.backward()
    optimizer_pre.step()

    if it % 1000 == 0:
        print(f"[Pretrain {it}] Data Loss: {loss_d.item():.6f}")

# Save pre-trained model
torch.save(model.state_dict(), "pretrained_model.pt")

# -----------------------------
# Fine-tuning: minimize data + PDE loss
# -----------------------------
print("\n Fine-tuning (data + PDE)...")

# Use L-BFGS for fine-tuning (second phase)
optimizer_fine = optim.LBFGS(model.parameters(), max_iter=finetune_iters, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

def closure():
    optimizer_fine.zero_grad()
    loss_d = data_loss(model, x_train, t_train, u_train)
    loss_p = pde_loss(model, x_colloc, t_colloc)
    total_loss = loss_d + loss_p
    total_loss.backward()
    return total_loss

optimizer_fine.step(closure)

# Final evaluation
final_loss_d = data_loss(model, x_train, t_train, u_train).item()
final_loss_p = pde_loss(model, x_colloc, t_colloc).item()
print(f"\n Final Data Loss: {final_loss_d:.6f}")
print(f" Final PDE Loss : {final_loss_p:.6f}")
print(f" Learned kappa  : {model.kappa.item():.6f}")
