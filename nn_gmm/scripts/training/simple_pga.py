"""Train a simple PGA with standard parameters (i.e. no location dependence)"""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

import ml_tools

# Grow the GPU memory usage as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
import nn_gmm

io_config = {
    "train_data_dirs": [
        "/home/claudy/dev/work/data/nn_gmm/training_data/train"
    ],
    "val_data_dirs": ["/home/claudy/dev/work/data/nn_gmm/training_data/val"],
    "stats_df": "/home/claudy/dev/work/data/nn_gmm/training_data/train/stats.csv",
    "base_output_dir": "/home/cbs51/dev/work/data/nn_gmm/results/test",

    "output_dir": None,
    "feature_config": {
        "dip": "min_max",
        "rake": "min_max",
        "mag": "standard",

        # "s": "standard",
        # "theta": "min_max",

        "vs30": "standard",
        # "z1p0": "standard",
        # "z2p5": "standard",
        # "vs500": "standard",

        "rrup": "standard",
        # "rx": "standard",
        # "rjb": "standard",
        # "ry": "standard",

        # "active_shallow": None,
        # "volcanic": None,
    },
    "im_config": {
        "PGA": None,
    },
}
stats_df = pd.read_csv(io_config["stats_df"], index_col="feature")
io_config["feature_config"] = nn_gmm.convert_pre_config(
    io_config["feature_config"], stats_df
)
io_config["im_config"] = nn_gmm.convert_pre_config(
    io_config["im_config"], stats_df
)

# Create the model
inputs = keras.Input(shape=len(io_config["feature_config"]))
x = ml_tools.hidden_layers.selu_dropout(inputs, 16, dropout=0.05)
x = ml_tools.hidden_layers.selu_dropout(x, 16, dropout=0.05)
outputs = keras.layers.Dense(1, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

train_config = {
    "batch_size": 1024,
    "shuffle_buffer_size": int(5e6),
    "n_epochs": 3,
    "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
    "loss": "mse",
    "use_sample_weights": False
}

# Create a run ID
run_id = nn_gmm.create_run_id()
io_config["run_id"] = run_id

train_result, output_dir, *_ = nn_gmm.train(io_config, train_config, model=model, verbose=1,)
