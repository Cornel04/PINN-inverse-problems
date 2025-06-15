import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        layer_list = []
        for i in range(len(layers) - 2):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            layer_list.append(nn.Tanh())  # or ReLU
        layer_list.append(nn.Linear(layers[-2], layers[-1]))  # output layer
        self.model = nn.Sequential(*layer_list)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.model(inputs)
