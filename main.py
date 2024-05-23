import os
import sys
import argparse
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Now import other modules that rely on this root directory
import config
from src.data.data_loader import load_data  # Assuming data_loader is a function in this module
from src.data.dataset import *
from src.utils.training_utils import *
from src.utils.evaluation_utils import *
from src.utils.prediction_utils import *


def predict(predict_data, args):
    source_file_path = "data/raw/globalTemperature/Land_and_Ocean_LatLong1.nc"
    new_file_path = "results/globalTemperature/temperature.nc"
    initialize_netcdf_with_historical_data(source_file_path=source_file_path, new_file_path=new_file_path)
    print("Making predictions")
    
    # input dimension is 6, 5 features and 1 target
    model = LSTMModel(input_dim=6, hidden_dim=100, num_layers=2)
    model.to(config.DEVICE)
    model.load_state_dict(torch.load("lstm_2_100_0.001_huber_4096_48.pt", map_location=config.DEVICE))
    model.eval()
    predict_and_update_nc_monthly(model, new_file_path, config.DEVICE, predict_data, start_year=2024, end_year=2025)
    

def main():
    parser = argparse.ArgumentParser(description="Run ML tasks such as training, evaluating, or predicting.")
    parser.add_argument("task", choices=["train", "evaluate", "expr", "predict"], help="Task to be performed.")
    parser.add_argument("--model", choices=["lstm", "rnn", "gru", "mlp", "attn"], default="lstm", help="Model type")
    parser.add_argument("--loss", choices=["mse", "mae", "huber"], type=str, default=config.LOSS, help="Loss function")
    parser.add_argument("--ssp", type=str, default="SSP119", help="Shared Socioeconomic Pathway, 'SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585'")
    parser.add_argument("--output", type=str, default="output.nc", help="Output data file")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=config.SEQUENCE_LENGTH, help="Sequence length")
    parser.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=config.NUM_LAYERS, help="Number of RNN layers")
    parser.add_argument("--epoch", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Patience for early stopping")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model")
    parser.add_argument("--load_model", action="store_true", help="Load the trained model")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Device to use for training")
    parser.add_argument("--save_result", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    # Load the data
    train_data, test_data, predict_data = load_data(args.ssp)

    if args.task == "train":
        train = train(train_data, test_data, args)
    elif args.task == "expr":
        expr(train_data, test_data, args)
    elif args.task == "evaluate":
        evaluate()
    elif args.task == "predict":
        predict(predict_data, args)
    else:
        print("Invalid task. Please choose from 'train', 'evaluate', 'expr' or 'predict'.")

if __name__ == "__main__":
    main()