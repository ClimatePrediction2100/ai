#!/bin/bash

# Define arrays for each hyperparameter
models=("lstm" "rnn" "gru" "mlp" "attn")
num_layers=(2 3 4)
hidden_dims=(50 100 200)
loss_functions=("mse" "mae" "huber")
batch_sizes=(32)  # Add your specific batch size values
epochs=(40)
patiences=(10)
learning_rates=(0.01 0.001 0.0001)
seq_lengths=(12 24 48 96)

# models=("lstm" "rnn")
# num_layers=(1)
# hidden_dims=(50)
# loss_functions=("mse")
# batch_sizes=(16)  # Add your specific batch size values
# epochs=(2)
# patiences=(10)
# learning_rates=(0.01)
# seq_lengths=(12)

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
                                    echo "Running with model=$model, num_layers=$num_layer, hidden_dim=$hidden_dim, loss=$loss, batch_size=$batch_size, epoch=$epoch, lr=$lr, seq_length=$seq_length, patience=$patience"
                                    python main.py "expr" --model $model --num_layers $num_layer --hidden_dim $hidden_dim --loss $loss --batch_size $batch_size --epoch $epoch --lr $lr --seq_length $seq_length --patience $patience
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done