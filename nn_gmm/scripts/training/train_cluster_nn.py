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
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

import ml_tools as mlt
from qcore import geo

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


def get_data(
    data_ds: tf.data.Dataset,
    feature_config: Dict,
    im_config: Dict,
    ims: Sequence[str],
    cluster_results: Dict,
    use_sample_weights: bool,
):
    # Convert feature config for pre-processing
    feature_config_dict, feature_config = (
        None,
        nn_gmm.convert_to_transform_fn(feature_config.copy()),
    )
    im_config = nn_gmm.convert_to_transform_fn(im_config.copy())

    feature_config["lon"] = None
    feature_config["lat"] = None

    # Pre-processing
    data_ds = nn_gmm.preprocess_ds(
        data_ds,
        feature_config=feature_config,
        feature_config_dict=feature_config_dict,
        im_config=im_config,
        use_sample_weights=use_sample_weights,
        as_dict=False,
    )

    tree = KDTree(
        cluster_results[ims[0]].loc[:, ["lon", "lat"]].values.astype(np.float32)
    )

    # Combine
    X = {f"{cur_im}_cluster_input": [] for cur_im in ims}
    X["inputs"] = []
    y = {cur_im: [] for cur_im in ims}
    sample_weights = [] if use_sample_weights else None
    for cur_batch in data_ds.as_numpy_iterator():
        batch_x = cur_batch[0]

        X["inputs"].append(batch_x[:, :-2])

        # Use distance to match stations
        dist, cluster_ind = tree.query(batch_x[:, -2:], 1)
        # assert np.all(np.isclose(dist, 0.0))

        # Add the cluster inputs & outputs
        for cur_im in ims:
            # Add outputs
            y[cur_im].append(cur_batch[1][cur_im])

            # Add clusters
            cur_cluster_df = cluster_results[cur_im]

            clusters = cur_cluster_df.cluster.values[cluster_ind]

            # Indices all have to be positive
            clusters += 1

            X[f"{cur_im}_cluster_input"].append(clusters.astype(int))

        if use_sample_weights:
            sample_weights.append(cur_batch[2])

    X = {cur_input: np.concatenate(cur_values) for cur_input, cur_values in X.items()}
    y = {cur_im: np.concatenate(cur_values) for cur_im, cur_values in y.items()}

    if sample_weights is not None:
        sample_weights = np.concatenate(sample_weights)
        return X, y, {cur_im: sample_weights for cur_im in ims}

    return X, y


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

    # Run the clustering
    cluster_results = nn_gmm.compute_clusters(
        base_model_dir,
        ims,
        eps=0.1,
        min_samples=2,
        bias_threshold=0.2,
        spatial_radius=12_000,
    )

    # Replace np.nan with -1 (noise cluster)
    for cur_im, cur_results in cluster_results.items():
        cur_results.cluster.loc[np.isnan(cur_results.cluster)] = -1.0

    # Not counting the noise cluster
    n_clusters = {
        cur_im: np.unique(cur_results.cluster).size - 1
        for cur_im, cur_results in cluster_results.items()
    }

    train_ds, val_ds = nn_gmm.load_datasets(
        nn_gmm.to_path(config["train_data_dirs"]),
        500_000,
        val_dirs=nn_gmm.to_path(config["val_data_dirs"]),
        shuffle_buffer_size=hyperparams["shuffle_buffer_size"],
        # shuffle_buffer_size=None,
        n_open_files=512,
    )

    train_data = get_data(
        train_ds,
        config["feature_config"].copy(),
        config["im_config"].copy(),
        ims,
        cluster_results,
        use_sample_weights=use_sample_weights,
    )

    val_data = get_data(
        val_ds,
        config["feature_config"].copy(),
        config["im_config"].copy(),
        ims,
        cluster_results,
        use_sample_weights=False,
    )

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
    l2 = hyperparams["l2"]

    n_layers = 2
    n_units = 8

    # Load the base model
    base_model = keras.models.load_model(base_model_dir / "best_model")
    base_model.trainable = False

    # Create the extra inputs
    cluster_inputs = {
        cur_im: keras.layers.Input(shape=(1,), name=f"{cur_im}_cluster_input",)
        for cur_im in ims
        if n_clusters[cur_im] > 0
    }

    # Rename the out layers
    # Only if cluster input is used
    for cur_layer in base_model.layers:
        if (
            "dense" not in cur_layer.name
            and "inputs" not in cur_layer.name
            and cur_layer._name in cluster_inputs.keys()
        ):
            cur_layer._name = f"{cur_layer.name}_base_out"

    outputs = []
    for cur_base_out in base_model.outputs:
        cur_im = cur_base_out.name.split("/")[0]

        if cur_im in cluster_inputs.keys():
            cur_input = cluster_inputs[cur_im]

            # Embedding layer
            # cur_x = mlt.hidden_layers.relu(cur_input, 10, name=f"{cur_im}_embedding")
            # cur_x = keras.layers.Embedding(
            #     input_dim=n_clusters[cur_im] + 1, output_dim=2
            # )(cur_input)
            # cur_x = keras.layers.Flatten()(cur_x)

            cur_x = keras.layers.CategoryEncoding(n_clusters[cur_im] + 1, output_mode="one_hot")(cur_input)

            cur_x = mlt.hidden_layers.relu(cur_x, 10, name=f"{cur_im}_embedding")

            # Concatenate
            cur_x = keras.layers.Concatenate()([cur_base_out, cur_x])

            # Dense layers
            for ix in range(n_layers):
                cur_x = mlt.hidden_layers.relu(
                    cur_x, n_units, l2=l2, name=f"{cur_im}_dense_{ix}"
                )

            outputs.append(
                keras.layers.Dense(1, activation=None, name=f"{cur_im}")(cur_x)
            )
        else:
            outputs.append(cur_base_out)

    model = keras.Model(
        inputs=list(cluster_inputs.values()) + [base_model.input], outputs=outputs
    )

    # Get the loss function
    hyperparams["loss"] = nn_gmm.get_loss_function(hyperparams)


    # log_dir = "/home/claudy/dev/work/tmp/logdir"
    # hyperparams["callbacks"] = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch='10, 15')]

    # Run training
    train_result = nn_gmm.run_training(
        model,
        train_data,
        val_data,
        config,
        hyperparams,
        nn_gmm.create_output_dir(config),
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


# class ClusterSequence(keras.utils.Sequence):
#     def __init__(
#         self,
#         batch_size: int,
#         data_ds: tf.data.Dataset,
#         feature_config: Dict,
#         im_config: Dict,
#         ims: Sequence[str],
#         cluster_results: Dict,
#         use_sample_weights: bool = True,
#     ):
#         self.batch_size = batch_size
#         self.cluster_results = cluster_results
#
#         self.ims = ims
#         self.feature_config = feature_config
#
#         self.data_ds = data_ds
#
#         # Convert feature config for pre-processing
#         feature_config_dict, feature_config = (
#             None,
#             nn_gmm.convert_to_transform_fn(feature_config.copy()),
#         )
#         im_config = nn_gmm.convert_to_transform_fn(im_config.copy())
#
#         feature_config["lon"] = None
#         feature_config["lat"] = None
#
#         # Pre-processing
#         data_ds = nn_gmm.preprocess_ds(
#             data_ds,
#             feature_config=feature_config,
#             feature_config_dict=feature_config_dict,
#             im_config=im_config,
#             use_sample_weights=use_sample_weights,
#             as_dict=False,
#         )
#
#         # Combine
#         self.X = []
#         self.y = {cur_im: [] for cur_im in ims}
#         self.sample_weights = [] if use_sample_weights else None
#         for cur_batch in data_ds.as_numpy_iterator():
#             self.X.append(cur_batch[0])
#
#             for cur_im, cur_values in cur_batch[1].items():
#                 self.y[cur_im].append(cur_values)
#
#             if use_sample_weights:
#                 self.sample_weights.append(cur_batch[2])
#
#         self.X = np.concatenate(self.X)
#         self.y = {
#             cur_im: np.concatenate(cur_values) for cur_im, cur_values in self.y.items()
#         }
#
#         if self.sample_weights is not None:
#             self.sample_weights = np.concatenate(self.sample_weights)
#
#         # Indices for shuffling
#         self.shuffle_ind = np.random.permutation(self.X.shape[0])
#
#     def __bool__(self):
#         return True
#
#     def __len__(self):
#         return int(np.ceil(self.X.shape[0] / self.batch_size))
#
#     def __getitem__(self, ix: int):
#         batch_x = self.X[self.shuffle_ind][
#             ix * self.batch_size : (ix + 1) * self.batch_size
#         ]
#
#         batch_inputs = {"inputs": batch_x[:, :-2]}
#         batch_outputs = {
#             cur_im: self.y[cur_im][self.shuffle_ind][
#                 ix * self.batch_size : (ix + 1) * self.batch_size
#             ]
#             for cur_im in self.ims
#         }
#
#         # Add the cluster inputs
#         cur_lon, cur_lat = batch_x[:, -2], batch_x[:, -1]
#         for cur_im in self.ims:
#             cur_cluster_df = self.cluster_results[cur_im]
#
#             # Use distance to find "merge"
#             dist_matrix = np.sqrt(
#                 (cur_lon[:, np.newaxis] - cur_cluster_df.lon.values) ** 2
#                 + (cur_lat[:, np.newaxis] - cur_cluster_df.lat.values) ** 2
#             )
#             clusters = cur_cluster_df.cluster.values[dist_matrix.argmin(axis=1)]
#             clusters[np.isnan(clusters)] = -1.0
#
#             # Indices all have to be positive
#             clusters += 1
#
#             batch_inputs[f"{cur_im}_cluster_input"] = clusters.astype(int)
#
#         if self.sample_weights is not None:
#             return (
#                 batch_inputs,
#                 batch_outputs,
#                 self.sample_weights[self.shuffle_ind][
#                     ix * self.batch_size : (ix + 1) * self.batch_size
#                 ],
#             )
#
#         return batch_inputs, batch_outputs
#
#     def on_epoch_end(self):
#         # Shuffle after each epoch
#         self.shuffle_ind = np.random.permutation(self.X.shape[0])
