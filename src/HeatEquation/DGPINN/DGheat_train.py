import torch
import torch.optim as optim
from DGheat_model import PINN
from DGheat_generate_data import generate_synthetic_data
from DGheat_loss import data_loss, pde_loss, initial_condition_loss, boundary_condition_loss
import pickle
import time

# Hyperparameters
N = 100
noise_std = 0.01
layers = [2, 50, 50, 50, 1]
pretrain_iters = 20000
finetune_iters = 10000
lr_pretrain = 1e-3

# Generate data
x_np, t_np, u_np = generate_synthetic_data(kappa=1.0, N=N, noise_std=noise_std)
x_train = torch.tensor(x_np, dtype=torch.float32)
t_train = torch.tensor(t_np, dtype=torch.float32)
u_train = torch.tensor(u_np, dtype=torch.float32)

# Collocation and condition points
x_colloc = x_train.clone().detach().requires_grad_(True)
t_colloc = t_train.clone().detach().requires_grad_(True)

x0 = x_train[t_train[:, 0] == 0]
t0 = t_train[t_train[:, 0] == 0]
u0 = u_train[t_train[:, 0] == 0]

xb0 = x_train[x_train[:, 0] == 0]
tb0 = t_train[x_train[:, 0] == 0]
ub0 = u_train[x_train[:, 0] == 0]

xb1 = x_train[x_train[:, 0] == 1]
tb1 = t_train[x_train[:, 0] == 1]
ub1 = u_train[x_train[:, 0] == 1]

xb = torch.cat([xb0, xb1], dim=0)
tb = torch.cat([tb0, tb1], dim=0)
ub = torch.cat([ub0, ub1], dim=0)


model = PINN(layers)  # Internally uses self.kappa_raw and def get_kappa(self): return torch.exp(self.kappa_raw)

# Pre-training
optimizer_pre = optim.Adam(model.parameters(), lr=lr_pretrain)

data_losses = []
pde_losses = []
kappa_values = []

print("\nPre-training (data only)...")
start_pretrain = time.time()
for it in range(pretrain_iters):
    optimizer_pre.zero_grad()
    loss_d = data_loss(model, x_train, t_train, u_train)
    loss_d.backward()
    data_losses.append(loss_d.item())
    pde_losses.append(0.0)  
    kappa_values.append(model.get_kappa().item())  # use log-space param

    optimizer_pre.step()
    if it % 1000 == 0:
        print(f"[Pretrain {it}] Data Loss: {loss_d.item():.6f}")

end_pretrain = time.time()
pretrain_duration = end_pretrain - start_pretrain
print(f"\nPre-training time: {pretrain_duration/60:.2f} minutes")
torch.save(model.state_dict(), "pretrained_model.pt")


# Fine-tuning
print("\nFine-tuning (data + PDE + IC + BC)...")
start_finetune = time.time()
optimizer_fine = optim.LBFGS(model.parameters(), max_iter=finetune_iters, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

λ_d, λ_p, λ_ic, λ_bc = 1.0, 1.0, 1.0, 1.0

def closure():
    optimizer_fine.zero_grad()
    loss_d = data_loss(model, x_train, t_train, u_train)
    loss_p = pde_loss(model, x_colloc, t_colloc)
    loss_ic = initial_condition_loss(model, x0, t0, u0)
    loss_bc = boundary_condition_loss(model, xb, tb, ub)
    total = λ_d * loss_d + λ_p * loss_p + λ_ic * loss_ic + λ_bc * loss_bc
    total.backward()

    # Track losses & kappa
    data_losses.append(loss_d.item())
    pde_losses.append(loss_p.item())
    kappa_values.append(model.get_kappa().item())  # use log-space param

    return total

optimizer_fine.step(closure)

end_finetune = time.time()
finetune_duration = end_finetune - start_finetune
print(f"Fine-tuning time: {finetune_duration/60:.2f} minutes")

# Final losses
final_d = data_loss(model, x_train, t_train, u_train).item()
final_p = pde_loss(model, x_colloc, t_colloc).item()
final_ic = initial_condition_loss(model, x0, t0, u0).item()
final_bc = boundary_condition_loss(model, xb, tb, ub).item()
print(f"\nFinal Losses:")
print(f"  Data     : {final_d:.6f}")
print(f"  PDE      : {final_p:.6f}")
print(f"  IC       : {final_ic:.6f}")
print(f"  BC       : {final_bc:.6f}")
print(f"  kappa    : {model.get_kappa().item():.6f}")

# Save logs
log_data = {
    "data_losses": data_losses,
    "pde_losses": pde_losses,
    "kappa_values": kappa_values
}

with open("training_logs.pkl", "wb") as f:
    pickle.dump(log_data, f)

print("\nTraining logs saved to training_logs.pkl")
