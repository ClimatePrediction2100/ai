import os
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the device to use for tensor computations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Model settings
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Feature settings, if any
FEATURE_DIM = 128

# Logging settings
LOGGING_CONFIG = {
    'version': 1,
    'handlers': {
        'stream_handler': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'detailed',
        },
    },
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'loggers': {
        'my_logger': {
            'handlers': ['stream_handler'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}

# Function to ensure all directories exist
def create_dirs():
    """Create directory structure if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)