import torch
import torch.nn as nn

from config import DEVICE as device


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # Forward pass through GRU layer
        out, _ = self.gru(x, h0)

        # Pass the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
