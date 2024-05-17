import torch
import torch.nn as nn

from config import DEVICE as device


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # Forward pass through RNN layer
        out, _ = self.rnn(x, h0)

        # Pass the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])

        # Squeeze the output to remove the last dimension if it's one
        out = out.squeeze(-1)
        return out
