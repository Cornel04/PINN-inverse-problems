import os
os.environ["DDE_BACKEND"] = "pytorch"  # Force DeepXDE to use PyTorch

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.backend import torch

# ------------------------
# 1. Generate Synthetic Data (Heat Equation Solution)
# ------------------------

def true_k(x):
    """ The actual unknown thermal conductivity we want to estimate. """
    return 0.5  # Real k value (we will see if PINN can learn this)

def heat_exact_solution(x, t):
    """ Exact solution of the heat equation (only for synthetic data generation). """
    return np.exp(-true_k(x) * np.pi**2 * t) * np.sin(np.pi * x)

# Generate synthetic data
x_vals = np.linspace(0, 1, 100)[:, None]
t_vals = np.linspace(0, 1, 100)[:, None]
X, T = np.meshgrid(x_vals, t_vals)
X_flat = X.flatten()[:, None]
T_flat = T.flatten()[:, None]

# Compute "measured" temperature values
u_real = heat_exact_solution(X_flat, T_flat)

# ------------------------
# 2. Define the Inverse Problem with PINNs
# ------------------------
def heat_equation(xt, u, k_pred):
    """ PINN formulation of the heat equation with an unknown k. """

    print(f"🔥 heat_equation called with xt.shape = {xt.shape}")  # Debugging

    if xt.shape[1] != 2:
        raise ValueError(f"Expected input `xt` to have shape (N,2) for (x,t), but got shape {xt.shape}")

    x = xt[:, 0:1]  # Extract x
    t = xt[:, 1:2]  # Extract t

    print(f"🔥 Extracted x.shape = {x.shape}, t.shape = {t.shape}")  # Debugging

    # Compute u_pred to ensure dependency on both x and t
    u_pred = model.net(xt)

    # 🔥 Explicitly extract `t` before computing `∂u/∂t`
    du_t = dde.grad.jacobian(u_pred, xt, i=1) if xt.shape[1] > 1 else dde.grad.jacobian(u_pred, xt, i=0)  
    du_xx = dde.grad.hessian(u_pred, xt, i=0, j=0)  # Compute second derivative w.r.t `x`

    return du_t - k_pred * du_xx  # Should be ≈ 0 when k is correct


# Geometry (1D spatial domain x ∈ [0,1], time t ∈ [0,1])
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geom_time = dde.geometry.GeometryXTime(geom, timedomain)

def boundary_func(x, on_boundary):
    x = np.atleast_2d(x)  # Ensure x is always 2D
    mask = np.isclose(x[:, 0], 0) | np.isclose(x[:, 0], 1)  # Detect x=0 or x=1 (boundaries)
    return np.logical_and(on_boundary, mask).reshape(on_boundary.shape)  # Ensure correct shape


bc = dde.icbc.DirichletBC(
    geom_time,
    lambda x: np.sin(np.pi * x[:, 0:1]),  # Ensure correct array shape
    boundary_func
)
ic = dde.icbc.IC(
    geom_time,
    lambda x: np.sin(np.pi * x[:, 0:1]),  # Ensure correct shape
    lambda x, on_initial: np.isclose(np.atleast_2d(x)[:, 1], 0).reshape(on_initial.shape)  # Ensures correct boolean shape
)








# Define unknown parameter (thermal conductivity k)
k_variable = dde.Variable(1.0)  # Initial guess for k

# ------------------------
# Debugging Boundary and Initial Condition Functions
# ------------------------

# # Test boundary function
# x_bc_test = np.array([[0, 0], [1, 0], [0.5, 0.5], [1, 1]])  # Example (x, t) points
# on_boundary_test = np.array([True, True, False, True])  # Manually define some boundary points
# print("\n🔍 Boundary function test input shape:", x_bc_test.shape)
# print("🔍 Boundary function test output:", boundary_func(x_bc_test, on_boundary_test))

# # Test initial condition
# x_ic_test = np.array([[0.2, 0], [0.5, 0], [0.8, 0], [0.5, 0.5]])  # Example (x, t) points
# print("\n🔍 Initial condition function test input shape:", x_ic_test.shape)
# print("🔍 Initial condition function test output:", np.isclose(np.atleast_2d(x_ic_test)[:, 1], 0))


# ------------------------
# Create the PDE problem
# ------------------------
data = dde.data.PDE(
    geom_time,
    lambda xt, u: heat_equation(xt, u, k_variable),  # Use fixed heat_equation function
    [bc, ic],
    num_domain=100,
    num_boundary=50
)


# ------------------------
# 3. Define and Train the PINN Model
# ------------------------

# Neural Network model
net = dde.nn.FNN([2, 50, 50, 50, 1], "tanh", "Glorot normal")

# Create the PINN model
model = dde.Model(data, net)

# Compile the model (we allow PINN to learn both k and the function u)
model.compile("adam", lr=0.001, external_trainable_variables=[k_variable])

# Train the model
losshistory, train_state = model.train(epochs=5000)

# ------------------------
# 4. Extract and Compare Estimated k
# ------------------------

k_estimated = k_variable.value.item()
print(f"\n🔍 Estimated k: {k_estimated}")
print(f"✅ True k: {true_k(0)}")

# ------------------------
# 5. Plot Training Loss
# ------------------------

plt.plot(losshistory.loss_train, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve for PINN Training")
plt.show()
