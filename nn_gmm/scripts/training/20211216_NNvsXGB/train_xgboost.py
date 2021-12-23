"""Train a XGBoost model"""
from pathlib import Path

import wandb
import tensorflow as tf
import pandas as pd
from wandb.xgboost import wandb_callback

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
    "train_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/training_data/train"],
    "val_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/training_data/val"],
    "stats_df": "/home/cbs51/dev/work/data/nn_gmm/training_data/train/stats.csv",
    "base_output_dir": "/home/cbs51/dev/work/data/nn_gmm/results/test",
    "basin_dir": "/home/cbs51/dev/work/data/nn_gmm/input_data/site_data/basin_stations",
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

    train_config = {
        "batch_size": 1024,
        "shuffle_buffer_size": int(5e6),
        "n_epochs": 15,
        "loss": "mse",
        "use_sample_weights": False,
        "cache": True,
    }

    model_params = {
        "tree_method": "approx",
        "subsample": 0.9,
        "disable_default_eval_metric": True,
    }

    if n_epochs is not None:
        train_config["n_epochs"] = n_epochs

    # Create a run ID
    run_id = nn_gmm.create_run_id()
    io_config["run_id"] = run_id

    if use_wandb:
        tags = list(io_config["im_config"].keys()) + ["XGB"]

        if "lat" in io_config["feature_config"].keys():
            tags.append("location")

        if train_config["use_sample_weights"]:
            tags.append("sample_weights")

        wandb.init(project="nn-gmm", entity="cbs51", tags=tags, name=run_id)

        wandb.config.input_config = io_config
        wandb.config.training_config = train_config

        if "callbacks" in train_config.keys():
            train_config["callbacks"].append(wandb_callback())
        else:
            train_config["callbacks"] = [wandb_callback()]

    # Train the model
    train_result = nn_gmm.train_xgb(io_config, train_config, model_params)
    output_dir = train_result.output_dir

    if eval:
        im = list(io_config["im_config"].keys())[0]

        # Write predictions
        nn_gmm.write_train_val_predictions(
            Path(io_config["train_data_dirs"][0]).parent, output_dir, verbose=False
        )

        # Print and compute general metrics
        train_metrics, val_metrics = nn_gmm.train_val_metrics(output_dir, save=True)

        # Print and compute basin metrics
        train_basin_metrics, val_basin_metrics = nn_gmm.train_val_basin_metrics(
            output_dir, Path(io_config["basin_dir"])
        )

        # Write metrics to wandb
        if use_wandb:
            ims = list(io_config["im_config"].keys())
            nn_gmm.wandb_log_metrics(wandb.run, ims, "train", train_metrics, train_basin_metrics)
            nn_gmm.wandb_log_metrics(wandb.run, ims, "val", val_metrics, val_basin_metrics)

        console.log("Generating binned Rrup plot")
        nn_gmm.gen_rrup_bin_plot(output_dir, im)

        console.log("Generating Rrup trend plot")
        nn_gmm.gen_rrup_trend_plot(output_dir, im)

        console.log("Generating residual plots")
        nn_gmm.gen_residual_plots(output_dir, im)


if __name__ == "__main__":
    typer.run(main)
