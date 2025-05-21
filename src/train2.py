# train2.py
import torch
import torch.optim as optim
from pinn_model2 import PINN_Wave
from generate_data2 import generate_synthetic_data_wave
from losses2 import data_loss, pde_loss_wave

# -----------------------------
# Hyperparameters
# -----------------------------
N = 100
noise_std = 0.01
layers = [2, 50, 50, 50, 1]
pretrain_iters = 20000
finetune_iters = 10000
lr_pretrain = 1e-3

# -----------------------------
# Data generation
# -----------------------------
x_np, t_np, u_np = generate_synthetic_data_wave(c=1.0, N=N, noise_std=noise_std)

x_train = torch.tensor(x_np, dtype=torch.float32)
t_train = torch.tensor(t_np, dtype=torch.float32)
u_train = torch.tensor(u_np, dtype=torch.float32)

x_colloc = x_train.clone().detach().requires_grad_(True)
t_colloc = t_train.clone().detach().requires_grad_(True)

# -----------------------------
# Initialize model
# -----------------------------
model = PINN_Wave(layers)

# -----------------------------
# Pre-training: data loss only
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
torch.save(model.state_dict(), "pretrained_model_wave.pt")

# -----------------------------
# Fine-tuning: data + PDE loss
# -----------------------------
print("\n Fine-tuning (data + PDE)...")

optimizer_fine = optim.LBFGS(model.parameters(), max_iter=finetune_iters, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

def closure():
    optimizer_fine.zero_grad()
    loss_d = data_loss(model, x_train, t_train, u_train)
    loss_p = pde_loss_wave(model, x_colloc, t_colloc)
    total_loss = loss_d + loss_p
    total_loss.backward()
    return total_loss

optimizer_fine.step(closure)

# Evaluation
final_loss_d = data_loss(model, x_train, t_train, u_train).item()
final_loss_p = pde_loss_wave(model, x_colloc, t_colloc).item()
print(f"\n Final Data Loss: {final_loss_d:.6f}")
print(f" Final PDE Loss : {final_loss_p:.6f}")
print(f" Learned wave speed c: {model.c.item():.6f}")
