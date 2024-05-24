#!/bin/bash

# Define the base directory for saving predictions and the path to model weights
SAVE_PATH_BASE="results/globalTemperature/predictions"
WEIGHT_PATH="results/models/lstm_2_100_0.01_mse_4096_1.pt"

# Define the range of years for prediction
START_YEAR=2024
END_YEAR=2150

# Array of SSP scenarios
SSP_SCENARIOS=("SSP119" "SSP126" "SSP245" "SSP370" "SSP434" "SSP460" "SSP534" "SSP585")

# Loop through each SSP scenario and run the prediction script
for SSP in "${SSP_SCENARIOS[@]}"; do
    echo "Running prediction for $SSP"
    SAVE_PATH="${SAVE_PATH_BASE}/${SSP}_predictions.nc"
    
    # Run the prediction Python script with the specified parameters
    python predict.py --ssp "$SSP" --save_path "$SAVE_PATH" --weight_path "$WEIGHT_PATH" --start_year $START_YEAR --end_year $END_YEAR
done

echo "All predictions completed."