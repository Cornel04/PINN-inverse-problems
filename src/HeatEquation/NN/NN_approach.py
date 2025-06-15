# nn_heat_equation.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)

# Create synthetic data
def generate_data(beta=1/20):
    x = np.linspace(0, 1, 201)
    t = np.linspace(0, 1, 201)
    X, T = np.meshgrid(x, t)
    U = np.exp(-(10 * np.pi * beta)**2 * T) * np.sin(10 * np.pi * X)
    return X.flatten()[:, None], T.flatten()[:, None], U.flatten()[:, None], X, T, U

# Define standard feedforward NN
class NN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=1):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

def train_nn(model, optimizer, x_train, t_train, u_train, epochs=5000):
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        u_pred = model(x_train, t_train)
        loss = loss_fn(u_pred, u_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4e}")
    return losses

def main():
    # Generate training data
    beta_true = 1 / 20
    x, t, u, X, T, U_true = generate_data(beta_true)
    x = torch.tensor(x, dtype=torch.float32).to(device)
    t = torch.tensor(t, dtype=torch.float32).to(device)
    u = torch.tensor(u, dtype=torch.float32).to(device)

    model = NN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _ = train_nn(model, optimizer, x, t, u, epochs=5000)

    # Predict with trained NN
    with torch.no_grad():
        u_pred = model(x, t).cpu().numpy()
    u_pred_2D = u_pred.reshape(201, 201)
    U_true_2D = U_true.reshape(201, 201)
    error_2D = np.abs(U_true_2D - u_pred_2D)

    # Plot true, predicted, and error
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].imshow(U_true_2D, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
    axs[0].set_title('Soluția analitică $u(x,t)$')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('t')

    axs[1].imshow(u_pred_2D, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
    axs[1].set_title('Predicția NN $u(x,t)$')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('t')

    im = axs[2].imshow(error_2D, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='hot')
    axs[2].set_title('Eroarea absolută $|u_{true} - u_{pred}|$')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('t')
    fig.colorbar(im, ax=axs[2])

    plt.suptitle("Comparație: soluție analitică, predicție NN și eroare")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
