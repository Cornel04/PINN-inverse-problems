import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
from pinn_bc_ic import PINN  
from pinn_bc_ic import generate_synthetic_data

# Load training logs
with open("training_logs_with_ic_bc.pkl", "rb") as f:
    logs = pickle.load(f)

# Plot loss curves
plt.figure()
plt.plot(logs["data_losses"], label="Data Loss")
plt.plot(logs["pde_losses"], label="PDE Loss")
plt.plot(logs["ic_losses"], label="IC Loss")
plt.plot(logs["bc_losses"], label="BC Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Training Losses Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Plot kappa evolution
plt.figure()
plt.plot(logs["kappa_values"])
plt.xlabel("Epoch")
plt.ylabel("kappa")
plt.title("Learned kappa over Epochs")
plt.grid(True)
plt.show()

# ---------- Optional: Visualize prediction vs true ----------
# Load model
layers = [2, 50, 50, 50, 1]
model = PINN(layers)
model.load_state_dict(torch.load("pinn_model_with_ic_bc.pt"))
model.eval()

# Generate test grid
x_np, t_np, u_true_np = generate_synthetic_data(kappa=1.0, N=100, noise_std=0.0)
x = torch.tensor(x_np, dtype=torch.float32)
t = torch.tensor(t_np, dtype=torch.float32)

# Predict
with torch.no_grad():
    u_pred = model(x, t).numpy()

# Plot prediction vs true
plt.figure(figsize=(10, 4))
plt.plot(u_true_np, label="True u(x, t)")
plt.plot(u_pred, label="Predicted u(x, t)", linestyle="--")
plt.title("Prediction vs Ground Truth")
plt.xlabel("Flattened (x,t) grid point")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.show()
