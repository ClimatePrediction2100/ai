import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_size=1):
        super(MLPModel, self).__init__()
        
        # Initialize layers using ModuleList
        self.layers = nn.ModuleList()
        
        # First layer from input dimension to the first hidden dimension
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Add hidden layers based on num_layers
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_size))

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # Pass through all hidden layers with ReLU activation, except the last layer
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        
        # No activation for the last layer (output layer)
        x = self.layers[-1](x)
        return x