#!/bin/bash

# Define arrays for each hyperparameter
models=("lstm" "rnn" "gru" "mlp" "attn")
num_layers=(2 4 6)
hidden_dims=(100 200)
loss_functions=("mse" "mae" "huber")
batch_sizes=(4096)
epochs=(40)
patiences=(10)
learning_rates=(0.01 0.001)
seq_lengths=(12 24 48)

# Calculate total number of experiments
total_experiments=$((${#models[@]} * ${#num_layers[@]} * ${#hidden_dims[@]} * ${#loss_functions[@]} * ${#batch_sizes[@]} * ${#epochs[@]} * ${#learning_rates[@]} * ${#seq_lengths[@]} * ${#patiences[@]}))
counter=0

# Iterate over each combination of hyperparameters
for model in "${models[@]}"; do
    for num_layer in "${num_layers[@]}"; do
        for hidden_dim in "${hidden_dims[@]}"; do
            for loss in "${loss_functions[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for epoch in "${epochs[@]}"; do
                        for lr in "${learning_rates[@]}"; do
                            for seq_length in "${seq_lengths[@]}"; do
                                for patience in "${patiences[@]}"; do
                                    ((counter++))
                                    echo "Running trial $counter/$total_experiments with model=$model, num_layers=$num_layer, hidden_dim=$hidden_dim, loss=$loss, batch_size=$batch_size, epoch=$epoch, lr=$lr, seq_length=$seq_length, patience=$patience"
                                    python main.py "expr" --model $model --num_layers $num_layer --hidden_dim $hidden_dim --loss $loss --batch_size $batch_size --epoch $epoch --lr $lr --seq_length $seq_length --patience $patience --save_model
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All experiments completed."