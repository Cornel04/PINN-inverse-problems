import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
        self.model = nn.ModuleList(layer_list)
        self.kappa = nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for i in range(len(self.model) - 1):
            inputs = self.activation(self.model[i](inputs))
        return self.model[-1](inputs)
