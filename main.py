import os
import sys

# Now import other modules that rely on this root directory
from src import config

def main():
    # Load the data
    train_data, test_data, predict_data = data_loader()

    # Initialize the model
    model = GRUModel(input_size=config.FEATURE_DIM, hidden_size=128, num_layers=2, output_size=1)

    # Train the model
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    train_model(model, train_loader, test_loader, num_epochs=config.NUM_EPOCHS, learning_rate=config.LEARNING_RATE)