"""Train a multi-output NN"""
import os
import shutil
import time
import argparse
from typing import List, Dict, Sequence
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


DATA_DIRS_LOOKUP = {
    "base": {
        "train_data_dirs": ["base_grid/train"],
        "val_data_dirs": ["base_grid/val"],
        "stats_df": "base_grid/train/stats.csv",
    },
    "base_fw": {
        "train_data_dirs": ["base-grid_fault-weighted/train"],
        "val_data_dirs": ["base-grid_fault-weighted/val"],
        "stats_df": "base-grid_fault-weighted/train/stats.csv",
    },
    # Fault station weighted normalised (with theta & s)
    "fswn_old": {
        "train_data_dirs": ["_fault_station_weighted_norm/train"],
        "val_data_dirs": ["_fault_station_weighted_norm/val"],
        "stats_df": "_fault_station_weighted_norm/train/stats.csv",
    },
    # Fault station weighted normalised
    "fswn": {
        "train_data_dirs": ["fault_station_weighted_norm/train"],
        "val_data_dirs": ["fault_station_weighted_norm/val"],
        "stats_df": "fault_station_weighted_norm/train/stats.csv",
    },
}


def run(
    machine_config_ffp: Path,
    io_config_ffp: Path,
    hyper_config_ffp: Path,
    data_key: str,
    use_wandb: bool = False,
    eval: bool = True,
    tags: List[str] = None,
    use_sample_weights: bool = False,
    early_stopping: bool = False,
    delete: bool = False,
    ims: Sequence[str] = None,
    args_hyperparams: Dict = None,
):
    tags = list(tags)

    machine_config = mlt.utils.load_yaml(machine_config_ffp)
    io_config = mlt.utils.load_yaml(io_config_ffp)
    hyperparams = mlt.utils.load_yaml(hyper_config_ffp)

    if args_hyperparams is not None:
        hyperparams = hyperparams | args_hyperparams

    if ims is not None:
        io_config["im_config"] = {
            cur_im: cur_value
            for cur_im, cur_value in io_config["im_config"].items()
            if cur_im in ims
        }

    # Add the data entries
    if data_key not in DATA_DIRS_LOOKUP.keys():
        raise ValueError(f"Invalid data key {data_key}")
    train_data_dir = machine_config["train_data_dir"]
    data_dirs = {
        cur_key: [os.path.join(train_data_dir, cur_dir) for cur_dir in cur_item]
        if isinstance(cur_item, list)
        else os.path.join(train_data_dir, cur_item)
        for cur_key, cur_item in DATA_DIRS_LOOKUP[data_key].items()
    }
    config = {**data_dirs, **io_config, **machine_config}

    # Prepare model input/output config for preprocesing
    stats_df = pd.read_csv(config["stats_df"], index_col="feature")
    config["feature_config"] = nn_gmm.convert_pre_config(
        config["feature_config"], stats_df
    )
    config["im_config"] = nn_gmm.convert_pre_config(config["im_config"], stats_df)

    # if n_epochs is not None:
    #     hyperparams["n_epochs"] = n_epochs

    if early_stopping:
        hyperparams["callbacks"] = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                min_delta=0.01 * len(config["im_config"]),
            )
        ]

    # Create a run ID
    run_id = nn_gmm.create_run_id(tags)
    config["run_id"] = run_id

    if use_sample_weights:
        tags.append("sample_weights")
    config["use_sample_weights"] = use_sample_weights

    if use_wandb:
        tags = [] if tags is None else list(tags)
        tags = tags + list(config["im_config"].keys()) + ["NN"]

        if (
            "lat" in config["feature_config"].keys()
            or "X" in config["feature_config"].keys()
        ):
            tags.append("location")

        wandb.init(
            project="nn-gmm",
            entity="cbs51",
            tags=tags,
            name=run_id,
            config=hyperparams,
        )
        print(hyperparams)
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
        console.print("Computing general metrics")
        train_metrics, val_metrics = nn_gmm.comp_train_val_metrics(
            output_dir, save=True, print_metrics=False
        )

        # Print and compute basin metrics
        console.print("Computing basin metrics")
        train_basin_metrics, val_basin_metrics = nn_gmm.comp_train_val_basin_metrics(
            output_dir, Path(config["basin_dir"]), save=True, print_metrics=False
        )

        # Compute spatial metrics
        console.print("Computing spatial metrics")
        nn_gmm.comp_train_val_spatial_metrics(output_dir, save=True)

        # Write metrics to wandb
        if use_wandb:
            nn_gmm.wandb_log_metrics(
                wandb.run, ims, "train", train_metrics, train_basin_metrics
            )
            nn_gmm.wandb_log_metrics(
                wandb.run, ims, "val", val_metrics, val_basin_metrics
            )

    if delete:
        console.print(f"Deleting output dir {output_dir}")
        shutil.rmtree(output_dir)

        # console.print("Generating binned Rrup plot")
        # nn_gmm.gen_rrup_bin_plot(output_dir, ims)

        # console.print("Generating residual plots")
        # nn_gmm.gen_residual_plots(output_dir, ims)

        # console.print("Generating Rrup trend plot")
        # nn_gmm.gen_rrup_trend_plots(output_dir, ims)

        # console.print("Generating spatial metric plots")
        # nn_gmm.gen_spatial_metric_plots(output_dir, ims, n_procs=14)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("machine_config_ffp", type=Path)
    parser.add_argument("io_config_ffp", type=Path)
    parser.add_argument("hyper_config_ffp", type=Path)
    parser.add_argument("data_key", type=str)
    parser.add_argument("--tags", nargs="+", type=str)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--use-sample-weights", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument(
        "--ims",
        default=None,
        type=str,
        help="Exclude any IMs not in this list",
        nargs="+",
    )

    # Hyperparameters (required for tuning via wandb sweep)
    # Using -1 as the default as None can be a valid value
    hyper_parser = argparse.ArgumentParser()
    hyper_parser.add_argument("--n_epochs", type=int, default=-1)
    hyper_parser.add_argument(
        "--hidden_layer_func", type=str, choices=["relu", "selu"], default="-1"
    )
    hyper_parser.add_argument("--l2", type=float, default=-1)
    hyper_parser.add_argument("--batch_size", type=int, default=-1)
    hyper_parser.add_argument("--learning_rate", type=float, default=-1)
    hyper_parser.add_argument("--n_units", type=int, default=-1)
    hyper_parser.add_argument("--n_layers", type=int, default=-1)
    hyper_parser.add_argument("--n_out_units", type=int, default=-1)
    hyper_parser.add_argument("--n_out_layers", type=int, default=-1)

    args, unknown_args = parser.parse_known_args()

    hyper_args = hyper_parser.parse_args(unknown_args)
    args_hyperparams = {
        cur_key: cur_item
        for cur_key, cur_item in vars(hyper_args).items()
        if (isinstance(cur_item, str) and cur_item != "-1")
        or (isinstance(cur_item, (int, float)) and int(cur_item) != -1)
    }

    run(
        args.machine_config_ffp,
        args.io_config_ffp,
        args.hyper_config_ffp,
        args.data_key,
        args.use_wandb,
        not args.no_eval,
        args.tags,
        args.use_sample_weights,
        args.early_stopping,
        args.delete,
        ims=args.ims,
        args_hyperparams=args_hyperparams,
    )
