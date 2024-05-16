import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super(MLPModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Forward pass through the first fully connected layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # Forward pass through the second fully connected layer
        x = self.fc4(x)
        return x
