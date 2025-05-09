"""Train a simple PGA"""
from pathlib import Path

import src.plotting.eval_plots
import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from wandb.keras import WandbCallback

import ml_tools
import typer

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
from nn_gmm import console

io_config = {
    "train_data_dirs": ["/home/claudy/dev/work/data/nn_gmm/training_data/train"],
    "val_data_dirs": ["/home/claudy/dev/work/data/nn_gmm/training_data/val"],
    "stats_df": "/home/claudy/dev/work/data/nn_gmm/training_data/train/stats.csv",
    "base_output_dir": "/home/cbs51/dev/work/data/nn_gmm/results/test",
    "basin_dir": "/home/claudy/dev/work/data/nn_gmm/input_data/site_data/basin_stations",
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
    "im_config": {"pSA_3.0": None,},
}


def main(use_wandb: bool = False, eval: bool = True, n_epochs: int = None):
    stats_df = pd.read_csv(io_config["stats_df"], index_col="feature")
    io_config["feature_config"] = nn_gmm.convert_pre_config(
        io_config["feature_config"], stats_df
    )
    io_config["im_config"] = nn_gmm.convert_pre_config(io_config["im_config"], stats_df)

    # Create the model
    inputs = keras.Input(shape=len(io_config["feature_config"]))
    x = ml_tools.hidden_layers.selu(inputs, 16, dropout=None)
    x = ml_tools.hidden_layers.selu(x, 16, dropout=None)
    outputs = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    train_config = {
        "batch_size": 1024,
        "shuffle_buffer_size": int(5e6),
        "n_epochs": 20,
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "loss": tf.losses.MeanSquaredError(),
        # "loss": nn_gmm.tf_mse,
        "use_sample_weights": False,
        "cache": True,
    }

    if n_epochs is not None:
        train_config["n_epochs"] = n_epochs

    # Create a run ID
    run_id = nn_gmm.create_run_id()
    io_config["run_id"] = run_id

    if use_wandb:
        tags = list(io_config["im_config"].keys()) + ["NN"]

        if "lat" in io_config["feature_config"].keys():
            tags.append("location")

        if train_config["use_sample_weights"]:
            tags.append("sample_weights")

        wandb.init(project="nn-gmm", entity="cbs51", tags=tags, name=run_id)

        wandb.config.input_config = io_config
        wandb.config.training_config = train_config

        if "callbacks" in train_config.keys():
            train_config["callbacks"].append(WandbCallback())
        else:
            train_config["callbacks"] = [WandbCallback()]

    train_result = nn_gmm.train_ds_nn(
        io_config, train_config, model=model, verbose=1
    )
    output_dir = train_result.output_dir

    # Write predictions
    if eval:
        im = list(io_config["im_config"].keys())[0]

        nn_gmm.write_train_val_predictions(
            Path(io_config["train_data_dirs"][0]).parent, output_dir, verbose=False
        )

        # Print and compute general metrics
        train_metrics, val_metrics = nn_gmm.comp_train_val_metrics(output_dir, save=True)

        # Print and compute basin metrics
        train_basin_metrics, val_basin_metrics = nn_gmm.comp_train_val_basin_metrics(
            output_dir, Path(io_config["basin_dir"])
        )

        # Write metrics to wandb
        if use_wandb:
            nn_gmm.wandb_log_metrics(wandb.run, [im], "train", train_metrics, train_basin_metrics)
            nn_gmm.wandb_log_metrics(wandb.run, [im], "val", val_metrics, val_basin_metrics)

        console.print("Generating binned Rrup plot")
        src.plotting.eval_plots.gen_rrup_bin_plots(output_dir, im)

        console.print("Generating Rrup trend plot")
        src.plotting.eval_plots.gen_rrup_trend_plots(output_dir, im)

        console.print("Generating residual plots")
        src.plotting.eval_plots.gen_residual_plots(output_dir, im)

if __name__ == "__main__":
    typer.run(main)
