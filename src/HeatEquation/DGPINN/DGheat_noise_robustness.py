import torch
import torch.optim as optim
from DGheat_model import PINN
from DGheat_generate_data import generate_synthetic_data
from DGheat_loss import data_loss, pde_loss, initial_condition_loss, boundary_condition_loss
import matplotlib.pyplot as plt
import pickle

# Hyperparameters
N = 100
layers = [2, 50, 50, 50, 1]
pretrain_iters = 5000
finetune_iters = 5000
lr_pretrain = 1e-3
noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
λ_d, λ_p, λ_ic, λ_bc = 1.0, 1.0, 1.0, 1.0

results = []

for noise_std in noise_levels:
    print(f"\n=== Testing noise_std = {noise_std} ===")

    # Generate noisy data
    x_np, t_np, u_np = generate_synthetic_data(kappa=1.0, N=N, noise_std=noise_std)
    x_train = torch.tensor(x_np, dtype=torch.float32)
    t_train = torch.tensor(t_np, dtype=torch.float32)
    u_train = torch.tensor(u_np, dtype=torch.float32)

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

    # Model
    model = PINN(layers)
    optimizer_pre = optim.Adam(model.parameters(), lr=lr_pretrain)

    # Pre-training
    for it in range(pretrain_iters):
        optimizer_pre.zero_grad()
        loss_d = data_loss(model, x_train, t_train, u_train)
        loss_d.backward()
        optimizer_pre.step()

    # Fine-tuning
    optimizer_fine = optim.LBFGS(model.parameters(), max_iter=finetune_iters, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        optimizer_fine.zero_grad()
        loss_d = data_loss(model, x_train, t_train, u_train)
        loss_p = pde_loss(model, x_colloc, t_colloc)
        loss_ic = initial_condition_loss(model, x0, t0, u0)
        loss_bc = boundary_condition_loss(model, xb, tb, ub)
        total = λ_d * loss_d + λ_p * loss_p + λ_ic * loss_ic + λ_bc * loss_bc
        total.backward()
        return total

    optimizer_fine.step(closure)

    # Record results
    final_kappa = model.get_kappa().item()
    final_data_loss = data_loss(model, x_train, t_train, u_train).item()
    final_pde_loss = pde_loss(model, x_colloc, t_colloc).item()

    print(f"  → Estimated κ: {final_kappa:.6f} | Data loss: {final_data_loss:.6e} | PDE loss: {final_pde_loss:.6e}")

    results.append({
        'noise_std': noise_std,
        'kappa': final_kappa,
        'data_loss': final_data_loss,
        'pde_loss': final_pde_loss
    })

# Save results
with open("noise_robustness_results.pkl", "wb") as f:
    pickle.dump(results, f)

# Plot results
noises = [r['noise_std'] for r in results]
kappas = [r['kappa'] for r in results]

plt.plot(noises, kappas, marker='o', label='Estimated kappa')
plt.axhline(y=1.0, color='r', linestyle='--', label='True kappa')
plt.xlabel("Noise standard deviation")
plt.ylabel("Estimated kappa")
plt.title("Noise Robustness of PINN")
plt.legend()
plt.grid(True)
plt.savefig("noise_robustness_plot.png")
plt.show()

print("\nFinished. Logs saved to 'noise_robustness_results.pkl'. Plot saved to 'noise_robustness_plot.png'.")
