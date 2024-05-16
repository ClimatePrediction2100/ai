import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

from config import DEVICE as device

def evaluate_model(model, test_loader, criterion=nn.MSELoss()):
    total_val_loss = 0
    count = 0
    actual = []
    predicted = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            val_outputs = model(xb)
            val_outputs = val_outputs.squeeze()  # Adjust output shape if necessary
            batch_loss = criterion(val_outputs, yb)
            total_val_loss += batch_loss.item()
            count += 1
            
            actual.extend(yb.tolist())
            predicted.extend(val_outputs.squeeze().tolist())

    avg_val_loss = total_val_loss / count
    print(f'Test Loss {avg_val_loss}')
    print(f'Actual mean: {np.mean(actual)}, Predicted mean: {np.mean(predicted)}')

    # Limit the data points to the first 100 for visualization
    actual = actual[200:260]
    predicted = predicted[200:260]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Data', color='blue')
    plt.plot(predicted, label='Predicted Data', color='red')
    plt.title('Comparison of Actual and Predicted Values for First 100 Time Steps')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (Normalized)')
    plt.legend()
    plt.show()