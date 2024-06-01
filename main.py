import argparse

# Now import other modules that rely on this root directory
import config
from src.data.data_loader import load_data  # Assuming data_loader is a function in this module
from src.utils.training_utils import train, expr
from src.utils.evaluation_utils import evaluate
from src.utils.prediction_utils import predict


def main():
    parser = argparse.ArgumentParser(description="Run ML tasks such as training, evaluating, or predicting.")
    parser.add_argument("task", choices=["train", "evaluate", "expr", "predict"], help="Task to be performed.")
    # train and evaluate arguments
    parser.add_argument("--model", choices=["lstm", "rnn", "gru", "mlp", "attn"], default="lstm", help="Model type")
    parser.add_argument("--loss", choices=["mse", "mae", "huber"], type=str, default=config.LOSS, help="Loss function")
    parser.add_argument("--output", type=str, default="output.nc", help="Output data file")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=config.SEQUENCE_LENGTH, help="Sequence length")
    parser.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=config.NUM_LAYERS, help="Number of RNN layers")
    parser.add_argument("--epoch", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Patience for early stopping")
    # predict arguments    
    parser.add_argument("--ssp", type=str, default="SSP119", help="Shared Socioeconomic Pathway, 'SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585'")
    parser.add_argument("--save_path", type=str, default="results/globalTemperature/predictions.nc", help="Path to save the prediction results")
    parser.add_argument("--weight_path", type=str, default="results/models/lstm_2_100_0.01_mse_4096_1.pt", help="Path to the trained model weights")
    parser.add_argument("--start_year", type=int, default=2024, help="Start year for prediction")
    parser.add_argument("--end_year", type=int, default=2150, help="End year for prediction")
    
    # Optional arguments
    parser.add_argument("--save_model", action="store_true", help="Save the trained model")
    parser.add_argument("--load_model", action="store_true", help="Load the trained model")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Device to use for training")
    parser.add_argument("--save_result", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    # Load the data
    train_data, test_data, predict_data = load_data(args.ssp)

    if args.task == "train":
        train(train_data, test_data, args)
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