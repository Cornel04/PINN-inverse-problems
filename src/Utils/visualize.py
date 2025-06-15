import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from normal_pinn_20k import PINN  # Make sure class definition matches your model
from normal_pinn_20k import generate_synthetic_data  # Use your same data generator

# ----- Load trained model -----
layers = [2, 50, 50, 50, 1]
model = PINN(layers)
model.load_state_dict(torch.load("pinn_model_nopretrain.pt"))
model.eval()

# ----- Load training logs -----
with open("training_logs_nopretrain.pkl", "rb") as f:
    logs = pickle.load(f)

data_losses = logs["data_losses"]
pde_losses = logs["pde_losses"]
kappa_values = logs["kappa_values"]

# ----- Generate ground truth -----
kappa_true = 1.0
N = 100
x_np, t_np, u_np = generate_synthetic_data(kappa=kappa_true, N=N, noise_std=0.0)

x = torch.tensor(x_np, dtype=torch.float32)
t = torch.tensor(t_np, dtype=torch.float32)
u_true = u_np.reshape(N, N)

# ----- Predict with model -----
with torch.no_grad():
    u_pred = model(x, t).cpu().numpy().reshape(N, N)

# ----- Plot: Predicted vs True -----
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(u_true, extent=[0, 1, 0, 1], origin="lower", aspect="auto", cmap="viridis")
plt.title("True Solution $u(x, t)$")
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(u_pred, extent=[0, 1, 0, 1], origin="lower", aspect="auto", cmap="viridis")
plt.title("Predicted Solution $\\hat{u}(x, t)$")
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar()

plt.subplot(1, 3, 3)
error = np.abs(u_true - u_pred)
plt.imshow(error, extent=[0, 1, 0, 1], origin="lower", aspect="auto", cmap="hot")
plt.title("Absolute Error $|u - \\hat{u}|$")
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar()

plt.tight_layout()
plt.savefig("pinn_nopretrain_u_comparison.png")
plt.show()

# ----- Plot: Loss Curves -----
plt.figure(figsize=(10, 4))
plt.semilogy(data_losses, label='Data Loss')
plt.semilogy(pde_losses, label='PDE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Loss Evolution During Training (No Pretrain)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pinn_nopretrain_loss_evolution.png")
plt.show()

# ----- Plot: Kappa Evolution -----
plt.figure(figsize=(8, 4))
plt.plot(kappa_values, label='Estimated κ (kappa)')
plt.axhline(y=kappa_true, color='r', linestyle='--', label='True κ')
plt.xlabel('Epoch')
plt.ylabel('κ value')
plt.title('Evolution of κ during Training (No Pretrain)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pinn_nopretrain_kappa_evolution.png")
plt.show()
