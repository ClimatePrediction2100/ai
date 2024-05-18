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
from src.utils.training_utils import train_model
from src.utils.evaluation_utils import evaluate_model
from src.model import *

def log_results(results, filename=os.path.join(config.ROOT_DIR, "results","experiment_results.csv")):
    fieldnames = list(results.keys())
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

def train(train_data, test_data, args):
    train_dataset = TrainData(train_data, args.seq_length)
    test_dataset = TestData(test_data, args.seq_length)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    
    feature_dim = train_dataset[0][0].shape[1]
    
    print(f"Model: {args.model}, Feature Dimension: {feature_dim}, Hidden Dimension: {args.hidden_dim}, Number of Layers: {args.num_layers}")
    if model == "lstm":
        model = LSTMModel(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif model == "gru":
        model = GRUModel(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif model == "rnn":
        model = RNNModel(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif model == "mlp":
        model = MLPModel(input_dim=feature_dim*args.seq_length, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif model == "attn":
        model = AttentionModel(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    
    model.to(config.DEVICE)
    
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    elif args.loss == "huber":
        criterion = nn.HuberLoss()
    
    model = train_model(model, train_loader, test_loader, args.epoch, args.device, save_model=args.save_model, \
        patience=args.patience, criterion=criterion, learning_rate=args.lr)
    
    return model


def expr(train_data, test_data, args):
    train_dataset = TrainData(train_data, args.seq_length)
    test_dataset = TestData(test_data, args.seq_length)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    
    feature_dim = train_dataset[0][0].shape[1]
    
    model = None
    if args.model == "lstm":
        model = LSTMModel(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.model == "gru":
        model = GRUModel(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.model == "rnn":
        model = RNNModel(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.model == "mlp":
        model = MLPModel(input_dim=feature_dim*args.seq_length, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.model == "attn":
        model = AttentionModel(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    
    model.to(args.device)
    
    criterion = None
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    elif args.loss == "huber":
        criterion = nn.HuberLoss()
    
    model= train_model(model, train_loader, test_loader, args.epoch, args.device, save_model=args.save_model,
        patience=args.patience, criterion=criterion, learning_rate=args.lr)
    
    avg_loss, actual_mean, predicted_mean = evaluate_model(model, test_loader, args.device, plot=False, criterion=criterion)
    
    # Log results to CSV
    results = {
        "model": args.model,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "avg_loss": avg_loss,
        "learning_rate": args.lr,
        "loss": args.loss,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "actual_mean": actual_mean,
        "predicted_mean": predicted_mean
    }
    log_results(results)
    
    if args.save_model:
        model_name = f"{args.model}_{args.num_layers}_{args.hidden_dim}_{args.lr}_{args.loss}_{args.batch_size}_{args.seq_length}" 
        torch.save(model.state_dict(), os.path.join(config.ROOT_DIR, "results", "models", f"{model_name}.pt"))

def evaluate(test_data, args, model):
    print("Evaluating the model")
    
    test_dataset = TestData(test_data, args.seq_length)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    elif args.loss == "huber":
        criterion = nn.HuberLoss()
    
    avg_loss, actual_mean, predicted_mean = evaluate_model(model, test_loader, args.device, plot=False, criterion=criterion)
    
    
def predict(predict_data, args):
    print("Making predictions")

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
        predict()
    else:
        print("Invalid task. Please choose from 'train', 'evaluate', 'expr' or 'predict'.")

if __name__ == "__main__":
    main()