import torch
import numpy as np
import matplotlib.pyplot as plt
from nn_model import MLP
from nn_generate import generate_synthetic_data

# Load model
layers = [2, 50, 50, 50, 1]
model = MLP(layers)
model.load_state_dict(torch.load("mlp_model.pt"))
model.eval()

# Generate test data
N = 100
x_np, t_np, u_true_np = generate_synthetic_data(kappa=1.0, N=N)
x_tensor = torch.tensor(x_np, dtype=torch.float32)
t_tensor = torch.tensor(t_np, dtype=torch.float32)

# Predict u(x, t)
with torch.no_grad():
    u_pred_tensor = model(x_tensor, t_tensor)
u_pred_np = u_pred_tensor.numpy()

# Reshape for plotting
X = x_np.reshape(N, N)
T = t_np.reshape(N, N)
U_true = u_true_np.reshape(N, N)
U_pred = u_pred_np.reshape(N, N)

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.contourf(X, T, U_true, 100, cmap='viridis')
plt.colorbar()
plt.title("True $u(x,t)$")

plt.subplot(1, 3, 2)
plt.contourf(X, T, U_pred, 100, cmap='viridis')
plt.colorbar()
plt.title("Predicted $u(x,t)$")

plt.subplot(1, 3, 3)
plt.contourf(X, T, np.abs(U_true - U_pred), 100, cmap='inferno')
plt.colorbar()
plt.title("Absolute Error")

plt.tight_layout()
plt.savefig("mlp_u_prediction_vs_true.png")
plt.show()
