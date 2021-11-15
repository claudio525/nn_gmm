from pathlib import Path
from typing import Dict
import os

import wandb
from wandb.keras import WandbCallback

import matplotlib

matplotlib.use("Agg")

import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras as keras

import nn_gmm


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

BASIN_DIR = Path(
    "/home/claudy/dev/work/data/nn_gmm/input_data/site_data/basin_stations"
)

IO_CONFIG = {
    "train_data_dirs": [
        "/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/train"
    ],
    "val_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/val"],
    "stats_df": "/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/train/stats.csv",
    "base_output_dir": "/home/cbs51/dev/work/data/nn_gmm/results/test",
    # "train_data_dirs": ["/mnt/win/image/clus/ml_data/nn_gmm/sample_files/train"],
    # "val_data_dirs": ["/mnt/win/image/clus/ml_data/nn_gmm/sample_files/val"],
    # "stats_df": "/mnt/win/image/clus/ml_data/nn_gmm/sample_files/train/stats.csv",
    # "base_output_dir": "/home/cbs51/code/nn_gmm/results/test",
    "output_dir": None,
    "multi_input": True,
    "feature_config": {
        "core_inputs": {
            "dip": "standard",
            "rake": "standard",
            # "strike": "standard",
            # "dtop": "standard",
            # "dbottom": "standard",
            # "ztor": "standard",
            # "zbot": "standard",
            "mag": "standard",
            "hdepth": "standard",
            # "hlon": "min_max",
            # "hlat": "min_max",
            # "width": "standard",
            # "length": "standard",
            # "is_point_source": None,
            # "lon": "min_max",
            # "lat": "min_max",
            "vs30": "standard",
            # "s": "standard",
            # "theta": "min_max",
            "z1p0": "standard",
            "z2p5": "standard",
            "vs500": "standard",
            "rrup": "standard",
            "rx": "standard",
            "rjb": "standard",
            "ry": "standard",
            "active_shallow": None,
            "volcanic": None,
        },
        "hydra_inputs": {
            "lon": "min_max",
            "lat": "min_max"
        },
    },
    "im_config": {
        "Ds595": "standard",
        "PGA": "standard",
        "pSA_0.5": "standard",
        "pSA_10.0": "standard",
        "pSA_5.0": "standard",
    },
}

stats_df = pd.read_csv(IO_CONFIG["stats_df"], index_col="feature")

IO_CONFIG = nn_gmm.convert_io_config(IO_CONFIG, stats_df)

CORE_IO_CONFIG = IO_CONFIG.copy()
CORE_IO_CONFIG["feature_config"] = CORE_IO_CONFIG["feature_config"]["core_inputs"]
CORE_IO_CONFIG["multi_input"] = False

# IO_CONFIG["feature_config"] = nn_gmm.convert_pre_config(
#     IO_CONFIG["feature_config"], stats_df
# )
# IO_CONFIG["im_config"] = nn_gmm.convert_pre_config(IO_CONFIG["im_config"], stats_df)


# Core Config
CORE_CONFIG = {
    "model_config": {
        "hidden_layer_config": {"dropout": 0.3},
        "hidden_layer_func": nn_gmm.relu_dropout,
        "units": [256, 256, 256],
    },
    "training_config": {
        "batch_size": 512,
        "shuffle_buffer_size": int(7e6),
        "n_epochs": 1,
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "loss": "mse",
        # "loss": nn_gmm.MargNLLLoss(len(list(INPUT_CONFIG["im_config"].keys()))),
        "callbacks": [keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6, min_delta=0.01)],
        "use_sample_weights": False,
    },
}

HYDRA_CONFIG = {
    "model_config": {
        "hidden_layer_config": {"dropout": 0.3},
        "hidden_layer_func": nn_gmm.relu_dropout,
        "output_units": [64, 64],
    },
    "training_config": {
        "batch_size": 512,
        "shuffle_buffer_size": int(7e6),
        "n_epochs": 1,
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "loss": "mse",
        # "loss": nn_gmm.MargNLLLoss(len(list(INPUT_CONFIG["im_config"].keys()))),
        # "callbacks": [],
        "callbacks": [keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6, min_delta=0.01)],
        "use_sample_weights": False,
    },
}

TUNING_TRAINING_CONFIG = {
        "batch_size": 512,
        "shuffle_buffer_size": int(7e3),
        "n_epochs": 1,
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.0001),
        "loss": "mse",
        "callbacks": [keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6, min_delta=0.01)],
        "use_sample_weights": False,
}


if __name__ == "__main__":
    use_wandb = False

    # Create a run ID
    run_id = nn_gmm.create_run_id()
    IO_CONFIG["run_id"], CORE_IO_CONFIG["run_id"] = run_id, run_id

    base_output_dir = nn_gmm.create_output_dir(IO_CONFIG)

    # # Train the core network
    # core_model_dir = base_output_dir / "core_model"
    # core_model_dir.mkdir(exist_ok=False, parents=False)
    # core_train_result, core_output_dir, *_ = nn_gmm.train(
    #     CORE_IO_CONFIG,
    #     CORE_CONFIG["training_config"],
    #     model_fn=nn_gmm.create_reg_model,
    #     model_config=CORE_CONFIG["model_config"],
    #     verbose=1,
    #     output_dir=core_model_dir,
    # )
    #
    # # Load the best model
    # core_model = keras.Sequential(
    #     keras.models.load_model(core_model_dir / "best_model").layers[:-2]
    # )
    # core_model.trainable = False
    #
    # # Add the subnetworks for the different IM outputs
    # hydra_model = nn_gmm.add_multi_output_head(
    #     core_model, "core_inputs", HYDRA_CONFIG["model_config"], list(IO_CONFIG["im_config"].keys()), hydra_input_t=("hydra_inputs", 2)
    # )
    #
    # # Train the hydra heads (keep the core model weights fixed
    # hydra_model_dir = base_output_dir / "hydra_model"
    # hydra_model_dir.mkdir(exist_ok=False, parents=False)
    # hydra_train_result, hydra_output_dir, *_ = nn_gmm.train(IO_CONFIG, HYDRA_CONFIG["training_config"], model=hydra_model,
    #                                                         model_config=HYDRA_CONFIG["model_config"], multi_output=True,
    #                                                         output_dir=hydra_model_dir)

    # Fine tuning
    if use_wandb:
        wandb.init(
            project="nn-gmm",
            entity="cbs51",
            tags=["location-dependence", "multi_output", "multi_input", loc_tag],
            name=run_id,
        )

        wandb.config.input_config = IO_CONFIG
        wandb.config.model_config = CONFIG["model_config"]
        wandb.config.training_config = CONFIG["training_config"]
        TUNING_TRAINING_CONFIG["callbacks"] = WandbCallback()

    # Load the model
    # model = keras.models.load_model(hydra_model_dir / "best_model")
    model = keras.models.load_model("/home/claudy/dev/work/data/nn_gmm/results/test/0127_2040/hydra_model/best_model")

    model_output_dir = base_output_dir / "complete_model"
    model_output_dir.mkdir()
    train_result, output_dir, *_ = nn_gmm.train(
        IO_CONFIG,
        TUNING_TRAINING_CONFIG,
        model,
        verbose=2,
        multi_output=True,
        output_dir=model_output_dir
    )

    # Location specific eval
    hydra_model = nn_gmm.GMM.load(train_result.best_model_dir)
    loc_vis_dir = output_dir / "visualisation" / "location"
    loc_vis_dir.mkdir(parents=True)
    basin_mean_abs_ln_res = nn_gmm.run_location_eval(
        hydra_model, Path(IO_CONFIG["val_data_dirs"][0]), BASIN_DIR, loc_vis_dir, suffix="val"
    )

    if use_wandb:
        # Upload the Basin residuals
        for cur_basin, cur_series in basin_mean_abs_ln_res.items():
            for cur_im, cur_value in cur_series.iteritems():
                wandb.run.summary[
                    f"mean_abs_ln_res_{cur_basin}_{cur_im.replace('.', 'p')}"
                ] = cur_value

        wandb.save(str(core_output_dir / "io_config.json"))
        wandb.save(str(core_output_dir / "model_config.json"))
        wandb.save(str(core_output_dir / "train_config.json"))

    exit()
