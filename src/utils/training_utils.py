import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import tqdm

import os
import config
from config import DEVICE as device

def train_model(model, train_loader, test_loader, num_epochs, patience=10, criterion=nn.MSELoss(), optimizer=optim.Adam, learning_rate=0.001):
    # Train the model
    total_step = len(train_loader)
    
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    model.to(device)
    
    min_loss = float('inf')
    model_path = os.path.join(config.ROOT_DIR, 'models', f'{model}')
    os.makedirs(model_path, exist_ok=True)  # Corrected from os.mkdir to os.makedirs
    
    epochs_no_improve = 0  # Counter to keep track of non-improving epochs
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= total_step
        
        # Validation phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {test_loss}')
        
        # Check if the loss improved
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pth'))
            model_name = f'{model}_epoch_{epoch+1}'
            print(f"Saved improved {model_name} with Test Loss: {test_loss}")
            epochs_no_improve = 0  # Reset the counter
        else:
            epochs_no_improve += 1  # Increment the counter
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Stopping training early - no improvement in {patience} epochs.")
            break

    return model