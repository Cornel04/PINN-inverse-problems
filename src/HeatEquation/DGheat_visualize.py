import torch
import numpy as np
import matplotlib.pyplot as plt
from DGheat_model import PINN
from HeatEquation.DGheat_generate_data import generate_synthetic_data

# Load model
layers = [2, 50, 50, 50, 1]
model = PINN(layers)
model.load_state_dict(torch.load("pretrained_model.pt"))
model.eval()

# Generate ground truth
kappa_true = 1.0
N = 100
x_np, t_np, u_np = generate_synthetic_data(kappa=kappa_true, N=N, noise_std=0.0)

x = torch.tensor(x_np, dtype=torch.float32)
t = torch.tensor(t_np, dtype=torch.float32)
u_true = u_np.reshape(N, N)

# Predict with model
with torch.no_grad():
    u_pred = model(x, t).cpu().numpy().reshape(N, N)

# Plot predicted vs true
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
plt.show()
