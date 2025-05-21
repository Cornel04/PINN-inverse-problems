# pinn_model2.py
import torch
import torch.nn as nn

class PINN_Wave(nn.Module):
    def __init__(self, layers):
        super(PINN_Wave, self).__init__()
        
        self.activation = nn.Tanh()
        
        # Define layers
        self.model = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)
        ])
        
        # Learnable wave speed c
        self.c = nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for i in range(len(self.model) - 1):
            inputs = self.activation(self.model[i](inputs))
        return self.model[-1](inputs)

if __name__ == "__main__":
    layers = [2, 50, 50, 50, 1]
    model = PINN_Wave(layers)
    x_test = torch.rand((10, 1))
    t_test = torch.rand((10, 1))
    u_pred = model(x_test, t_test)
    print(f"Predicted u shape: {u_pred.shape}")
    print(f"Learnable wave speed c: {model.c.item()}")
