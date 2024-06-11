
# Climate Prediction AI Model Environment Setup Tutorial

This tutorial guides you through the process of setting up the project environment for the Climate Prediction AI Model. Follow the steps below to configure your system, download necessary data, and prepare for running simulations.

## Prerequisites

Before you begin, ensure you have the following installed on your machine:
- `bash` - For running shell scripts.
- `wget` - For downloading files from the internet.
- `pip` - For installing Python packages.
- `tar` - For extracting tar files.

## Project Structure

The Makefile manages several directories and tasks including data downloading, environment setup, and simulation execution. Here's a brief outline of the project's directory structure:

- `data/raw/` - Directory for storing raw data files.
- `results/` - Directory for storing output from simulations and model weights.

## Setup Steps

### 1. Clone the Repository

Start by cloning the repository to your local machine. Replace `<url-to-repository>` with the actual URL of your repository:

```bash
git clone https://github.com/ClimatePrediction2100/ai.git
cd ai
```

### 2. Run Setup

The setup process involves creating necessary directories and installing required Python packages. Run the following command:

```bash
make setup
```

This command performs the following actions:
- Creates directories for storing data and results.
- Installs Python dependencies listed in `requirements.txt`.

### 3. Download Data

To download the required datasets for the project, execute:

```bash
make download_data
```

This command downloads various climate data files needed for simulation into the `data/raw/` directory.

### 4. Download Model Weights (Optional)

If pre-trained model weights are required for the simulation, use:

```bash
make download_weights
```

This command downloads and extracts the model weights into the `results/` directory.

### 5. Download Simulation Results (Optional)

To download pre-computed simulation results, run:

```bash
make download_results
```

This command downloads and extracts result files into the `results/` directory.

## Hyperparameter Testing

I have run hyperparameter tuning experiments to find the best hyperparameters for the AI model. The results are stored in the `results/` directory. You can use these results to select the best hyperparameters for your model.

You can change the hyperparameters by modifying 'expr.sh' script. The script is used to run the AI model experiments.

### Experiment Execution

To perform a new experiment, run:

```bash
make expr
```

This will execute the `expr.sh` script, which is configured to run your AI model experiments.

### Simulation

You can modify the scenarios, model weights, and other configurations in the `predict.sh` script.

To run prediction simulations, use:

```bash
make simulate
```

This will execute the `predict.sh` script, which likely runs simulations based on the AI model's predictions.

## Conclusion

By following this tutorial, you should now have a fully set up environment ready for performing climate prediction simulations using AI models. This setup ensures that all the necessary data and configurations are in place for effective and reproducible research.
