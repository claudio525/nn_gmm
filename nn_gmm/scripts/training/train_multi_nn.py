"""Train a multi-output NN"""
import time
from typing import List
from pathlib import Path

import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from wandb.keras import WandbCallback

import ml_tools as mlt
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

app = typer.Typer()

DATA_DIRS_LOOKUP = {
    "base": {
        "train_data_dirs": [
            "/home/claudy/dev/work/data/nn_gmm/training_data/base_grid/train"
        ],
        "val_data_dirs": [
            "/home/claudy/dev/work/data/nn_gmm/training_data/base_grid/val"
        ],
        "stats_df": "/home/claudy/dev/work/data/nn_gmm/training_data/base_grid/train/stats.csv",
    },
    "base_fw": {
        "train_data_dirs": [
            "/home/claudy/dev/work/data/nn_gmm/training_data/base-grid_fault-weighted/train"
        ],
        "val_data_dirs": [
            "/home/claudy/dev/work/data/nn_gmm/training_data/base-grid_fault-weighted/val"
        ],
        "stats_df": "/home/claudy/dev/work/data/nn_gmm/training_data/base-grid_fault-weighted/train/stats.csv",
    },
    "fswn": {
        "train_data_dirs": ["/home/claudy/dev/work/data/nn_gmm/training_data/fault_station_weighted_norm/train"],
        "val_data_dirs": ["/home/claudy/dev/work/data/nn_gmm/training_data/fault_station_weighted_norm/val"],
        "stats_df": "/home/claudy/dev/work/data/nn_gmm/training_data/fault_station_weighted_norm/train/stats.csv",
    },
}


@app.command("run-config")
def run_config(config_ffp: Path, eval: bool = True):
    print(f"wtf")


@app.command("run")
def run(
    machine_config_ffp: Path,
    io_config_ffp: Path,
    hyper_config_ffp: Path,
    data_key: str,
    use_wandb: bool = False,
    eval: bool = True,
    n_epochs: int = None,
    tags: List[str] = None,
    use_sample_weights: bool = False,
):
    tags = list(tags)

    machine_config = mlt.utils.load_yaml(machine_config_ffp)
    io_config = mlt.utils.load_yaml(io_config_ffp)
    hyperparams = mlt.utils.load_yaml(hyper_config_ffp)

    # Add the data entries
    if data_key not in DATA_DIRS_LOOKUP.keys():
        raise ValueError(f"Invalid data key {data_key}")
    config = {**DATA_DIRS_LOOKUP[data_key], **io_config, **machine_config}

    # Prepare model input/output config for preprocesing
    stats_df = pd.read_csv(config["stats_df"], index_col="feature")
    config["feature_config"] = nn_gmm.convert_pre_config(
        config["feature_config"], stats_df
    )
    config["im_config"] = nn_gmm.convert_pre_config(config["im_config"], stats_df)

    if n_epochs is not None:
        hyperparams["n_epochs"] = n_epochs

    # Create a run ID
    run_id = nn_gmm.create_run_id(tags)
    config["run_id"] = run_id

    if use_sample_weights:
        tags.append("sample_weights")
        config["use_sample_weights"] = use_sample_weights

    if use_wandb:
        tags = [] if tags is None else list(tags)
        tags = tags + list(config["im_config"].keys()) + ["NN"]

        if "lat" in config["feature_config"].keys():
            tags.append("location")

        wandb.init(
            project="nn-gmm",
            entity="cbs51",
            tags=tags,
            name=run_id,
            config=hyperparams,
        )

        wandb.config["use_sample_weights"] = True
        wandb.config.config = config

        if "callbacks" in hyperparams.keys():
            hyperparams["callbacks"].append(WandbCallback())
        else:
            hyperparams["callbacks"] = [WandbCallback()]

    # Create the model
    model = nn_gmm.create_reg_multi_output_model(
        hyperparams, len(config["feature_config"]), list(config["im_config"].keys())
    )

    # Get the loss function
    hyperparams["loss"] = nn_gmm.get_loss_function(hyperparams)

    # Run training
    train_result = nn_gmm.train_nn(
        config, hyperparams, model=model, multi_output=False, verbose=2,
    )
    output_dir = train_result.output_dir

    # Write predictions
    if eval:
        ims = list(io_config["im_config"].keys())

        start_time = time.time()
        nn_gmm.write_train_val_predictions(
            Path(config["train_data_dirs"][0]).parent,
            output_dir,
            verbose=False,
            batch_size=1_000_000,
        )
        console.print(f"Took {time.time() - start_time}s to get predictions")

        # Print and compute general metrics
        train_metrics, val_metrics = nn_gmm.comp_train_val_metrics(
            output_dir, save=True
        )

        # Print and compute basin metrics
        train_basin_metrics, val_basin_metrics = nn_gmm.comp_train_val_basin_metrics(
            output_dir, Path(config["basin_dir"], save=True)
        )

        # Compute spatial metrics
        nn_gmm.comp_train_val_spatial_metrics(output_dir, save=True)

        # Write metrics to wandb
        if use_wandb:
            nn_gmm.wandb_log_metrics(
                wandb.run, ims, "train", train_metrics, train_basin_metrics
            )
            nn_gmm.wandb_log_metrics(
                wandb.run, ims, "val", val_metrics, val_basin_metrics
            )

        # console.print("Generating binned Rrup plot")
        # nn_gmm.gen_rrup_bin_plot(output_dir, ims)

        # console.print("Generating residual plots")
        # nn_gmm.gen_residual_plots(output_dir, ims)

        # console.print("Generating Rrup trend plot")
        # nn_gmm.gen_rrup_trend_plots(output_dir, ims)

        # console.print("Generating spatial metric plots")
        # nn_gmm.gen_spatial_metric_plots(output_dir, ims, n_procs=14)


if __name__ == "__main__":
    app()
