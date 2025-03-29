# pinn_model.py
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        
        # Activation function
        self.activation = nn.Tanh()
        
        # Define neural network layers
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
        
        # Combine layers into a Sequential module
        self.model = nn.ModuleList(layer_list)
        
        # Learnable parameter kappa (κ), initialized randomly
        self.kappa = nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, x, t):
        # Combine inputs x and t into one tensor
        inputs = torch.cat([x, t], dim=1)
        
        # Pass inputs through hidden layers with activation
        for i in range(len(self.model) - 1):
            inputs = self.activation(self.model[i](inputs))
        
        # Output layer without activation
        output = self.model[-1](inputs)
        return output

# Example usage
if __name__ == "__main__":
    # Example architecture: 2 inputs (x, t), 3 hidden layers (50 neurons each), 1 output (u)
    layers = [2, 50, 50, 50, 1]
    model = PINN(layers)
    
    # Test the model with random data
    x_test = torch.rand((10, 1))
    t_test = torch.rand((10, 1))
    u_pred = model(x_test, t_test)
    
    print(f"Predicted u shape: {u_pred.shape}")
    print(f"Learnable kappa: {model.kappa.item()}")
