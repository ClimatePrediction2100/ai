import torch
import torch.nn as nn
import math

from config import DEVICE as device


class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_size=1):
        super(AttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Compute attention weights
        energy = torch.tanh(self.attn(out))
        attention = torch.softmax(torch.matmul(energy, self.v), dim=1)

        attention = attention.unsqueeze(-1)

        # Apply attention weights to LSTM outputs
        context = attention * out
        context = torch.sum(context, dim=1)

        # Pass the context to the fully connected layer
        out = self.fc(context)
        return out
