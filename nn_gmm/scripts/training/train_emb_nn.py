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


def run(
        base_model_dir: Path,
        hyper_config_ffp: Path,
        use_wandb: bool = False,
        eval: bool = True,
        tags: List[str] = None,
        use_sample_weights: bool = False,
        delete: bool = False,
        args_hyperparams: Dict = None,
):
    tags = list(tags) if tags is not None else []

    config = mlt.utils.load_json(base_model_dir / "input_config.json")
    hyperparams = mlt.utils.load_yaml(hyper_config_ffp)

    # Save base model used
    config["base_model_dir"] = str(base_model_dir)

    # Get list of output IMs
    ims = list(config["im_config"].keys())

    if args_hyperparams is not None:
        hyperparams = hyperparams | args_hyperparams

    # Create a run ID
    run_id = nn_gmm.create_run_id(tags)
    config["run_id"] = run_id

    if use_sample_weights:
        tags.append("sample_weights")
    config["use_sample_weights"] = use_sample_weights

    # Update config to be multi-input
    config["feature_config"] = {"inputs": config["feature_config"]}
    config["feature_config"]["site_inputs"] = nn_gmm.convert_pre_config(
        {"site": None},
        pd.read_csv(config["stats_df"], index_col="feature"),
    )
    config["feature_config"]["meta_inputs"] = nn_gmm.convert_pre_config(
        {"mag": "standard"},
        pd.read_csv(config["stats_df"], index_col="feature"),
    )

    config["multi_input"] = True

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
            hyperparams["callbacks"].append(WandbCallback(save_model=False))
        else:
            hyperparams["callbacks"] = [WandbCallback(save_model=False)]

    # Create site lookup
    data_df = nn_gmm.load_dataset_as_df(Path(config["train_data_dirs"][0]), ["site"])

    unique_sites = np.unique(data_df.site.values)
    site_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            unique_sites, tf.range(unique_sites.size, dtype=tf.int64)
        ),
        default_value=-1,
    )

    # Create the model
    n_pre_layers = hyperparams["n_pre_layers"]
    n_pre_units = hyperparams["n_pre_units"]

    n_core_layers = hyperparams["n_core_layers"]
    n_core_units = hyperparams["n_core_units"]

    n_out_layers = hyperparams["n_out_layers"]
    n_out_units = hyperparams["n_out_units"]

    # Load the base model
    base_model = keras.models.load_model(base_model_dir / "best_model")
    base_model.trainable = False
    # base_input = base_model.layers[0]
    #
    # # Get the base dense layers
    # x_base = None
    # for cur_layer in base_model.layers:
    #     if "dense" in cur_layer.name:
    #         cur_layer.trainable = False
    #         x_base = cur_layer

    # Rename the out layers
    for cur_layer in base_model.layers:
        if "dense" not in cur_layer.name and "inputs" not in cur_layer.name:
            cur_layer._name = f"{cur_layer.name}_base"

    # Add the pre location layers
    x_sites = site_input = keras.layers.Input(
        len(config["feature_config"]["site_inputs"]), name="site_inputs", dtype=tf.string
    )
    x_site_ind = keras.layers.Lambda(lambda sites: site_table.lookup(sites))(x_sites)
    x_emb = keras.layers.Embedding(input_dim=unique_sites.size, output_dim=4)(x_site_ind)
    x_emb = keras.layers.Flatten()(x_emb)

    # for ix in range(n_pre_layers):
    #     x_emb = mlt.hidden_layers.relu(x_emb, n_pre_units, name=f"loc_pre_dense_{ix}")

    # No core layers
    if n_core_layers == 0:
        outputs = []
        for cur_base_out in base_model.outputs:
            # Hydra layers
            cur_x = keras.layers.Concatenate()([cur_base_out, x_emb])
            if n_out_layers > 0:
                for ix in range(n_out_layers):
                    cur_x = mlt.hidden_layers.relu(
                        cur_x,
                        n_out_units,
                        name=f"{cur_base_out.name.split('/')[0]}_dense_{ix}",
                    )

            outputs.append(
                keras.layers.Dense(
                    1, activation=None, name=f"{cur_base_out.name.split('/')[0]}"
                )(cur_x)
            )
    else:
        x_base_output = keras.layers.Concatenate()(base_model.outputs)

        # Concatenate
        x = keras.layers.Concatenate()([x_emb, x_base_output])

        # Core layers
        for ix in range(n_core_layers):
            x = mlt.hidden_layers.relu(x, n_core_units, name=f"loc_core_dense_{ix}")

        # Hydra layers
        outputs = []
        for cur_im in ims:
            cur_x = x
            if n_out_layers > 0:
                for ix in range(n_out_layers):
                    cur_x = mlt.hidden_layers.relu(
                        cur_x, n_out_units, name=f"{cur_im}_dense_{ix}"
                    )

            outputs.append(
                keras.layers.Dense(1, activation=None, name=f"{cur_im}")(cur_x)
            )

    model = keras.Model(inputs=[site_input, base_model.input], outputs=outputs)

    # Get the loss function
    hyperparams["loss"] = nn_gmm.get_loss_function(hyperparams)

    # loss_weights = {"PGA": 1.0, "PGV": 1.0,
    #                 "pSA_0.5": 1.0, "pSA_0.25": 1.0,
    #                 "pSA_1.0": 1.0, "pSA_3.0": 3.0,
    #                 "pSA_5.0": 3.0, "pSA_10.0": 3.0}

    # Run training
    train_result = nn_gmm.train_ds_nn(
        config, hyperparams, model=model, as_dict=True, verbose=2
    )
    output_dir = train_result.output_dir

    # Write predictions
    if eval:
        ims = list(config["im_config"].keys())

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
        nn_gmm.comp_train_val_spatial_metrics(
            output_dir, save=True, use_sample_weights=use_sample_weights
        )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("base_model_dir", type=Path, help="Path to the base model")
    parser.add_argument("hyper_config_ffp", type=Path)
    parser.add_argument("--tags", nargs="+", type=str)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--use-sample-weights", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--delete", action="store_true")

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
    hyper_parser.add_argument("--n_pre_units", type=int, default=-1)
    hyper_parser.add_argument("--n_pre_layers", type=int, default=-1)
    hyper_parser.add_argument("--n_core_units", type=int, default=-1)
    hyper_parser.add_argument("--n_core_layers", type=int, default=-1)
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
        args.base_model_dir,
        args.hyper_config_ffp,
        args.use_wandb,
        not args.no_eval,
        args.tags,
        args.use_sample_weights,
        args.delete,
        args_hyperparams=args_hyperparams,
    )
