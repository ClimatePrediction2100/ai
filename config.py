import os
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the device to use for tensor computations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
NUM_WORKERS = 16 if DEVICE == 'cuda' else 8 if DEVICE == 'mps' else 0

# Model settings
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_LAYERS = 4
HIDDEN_DIM = 512
LOSS = 'mse'
PATIENCE = 10
SEQUENCE_LENGTH = 12

DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'results', 'globalTemperature')

# Function to ensure all directories exist
def create_dirs():
    """Create directory structure if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)