import torch
import torch.nn as nn
import torch.optim as optim
from nn_model import MLP
from nn_generate import generate_synthetic_data

# Hyperparameters
N = 100
noise_std = 0.01
layers = [2, 50, 50, 50, 1]
num_epochs = 5000
learning_rate = 1e-3

# Generate data
x_np, t_np, u_np = generate_synthetic_data(kappa=1.0, N=N, noise_std=noise_std)

x_train = torch.tensor(x_np, dtype=torch.float32)
t_train = torch.tensor(t_np, dtype=torch.float32)
u_train = torch.tensor(u_np, dtype=torch.float32)

# Initialize model
model = MLP(layers)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    u_pred = model(x_train, t_train)
    loss = criterion(u_pred, u_train)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Save model
torch.save(model.state_dict(), "mlp_model.pt")
