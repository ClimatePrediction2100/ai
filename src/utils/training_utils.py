import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import config
import csv

from src.utils.evaluation_utils import evaluate_model
from src.data.data_loader import load_data  # Assuming data_loader is a function in this module
from src.data.dataset import *
from src.model import *

def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    device,
    save_model=False,
    patience=10,
    criterion=nn.MSELoss(),
    optimizer=optim.Adam,
    learning_rate=0.001,
):
    # Train the model
    total_step = len(train_loader)

    optimizer = optimizer(model.parameters(), lr=learning_rate)
    model.to(device)

    min_loss = float("inf")
    model_path = os.path.join(config.ROOT_DIR, "models", f"{model}")
    best_model_state = None  # To store the state of the best model

    # if save_model:
    #     os.makedirs(model_path, exist_ok=True)

    epochs_no_improve = 0

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

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}"
        )

        # Check if the loss improved
        if test_loss < min_loss:
            min_loss = test_loss
            best_model_state = model.state_dict()  # Save the best model state
            # if save_model:
            #     torch.save(best_model_state, os.path.join(model_path, "best_model.pth"))
            model_name = f"{model}_epoch_{epoch+1}"
            print(f"Improved {model_name} with Test Loss: {test_loss}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Stopping training early - no improvement in {patience} epochs.")
            break

    # Restore the best model before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


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
