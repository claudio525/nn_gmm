from pathlib import Path
from typing import Dict
import os

import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
import numpy as np
import pandas as pd

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


# TRAIN_REP_EVENTS = ['3369048', '3468980', '2626467', 'Tuakana10', 'MS05', 'Wairau']
# VAL_REP_EVENTS = ['2937330', '3366544', '3367749', 'Orakonui', 'TuhuaS01', 'HopeConway']


INPUT_CONFIG = {
    "train_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/train"],
    "val_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/val"],
    "stats_df": "/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/train/stats.csv",
    "base_output_dir": "/home/cbs51/dev/work/data/nn_gmm/results/test",

    # "train_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/small_mod_events_sample_files/train"],
    # "val_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/small_mod_events_sample_files/val"],
    # "stats_df": "/home/cbs51/dev/work/data/nn_gmm/input_data/small_mod_events_sample_files/train/stats.csv",
    # "base_output_dir": "/home/cbs51/dev/work/data/nn_gmm/results/test",

    # "train_data_dirs": ["/mnt/win/image/clus/ml_data/nn_gmm/sample_files/train"],
    # "val_data_dirs": ["/mnt/win/image/clus/ml_data/nn_gmm/sample_files/val"],
    # "stats_df": "/mnt/win/image/clus/ml_data/nn_gmm/sample_files/train/stats.csv",
    # "base_output_dir": "/home/cbs51/code/nn_gmm/results/test",

    "output_dir": None,
    "feature_config": {
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
        "width": "standard",
        "length": "standard",
        # "is_point_source": None,
        "lon": "min_max",
        "lat": "min_max",
        "vs30": "standard",
        "s": "standard",
        "theta": "min_max",

        "z1p0": "standard",
        "z2p5": "standard",
        "vs500": "standard",
        # "rrup": "standard",
        "rx": "standard",
        "rjb": "standard",
        "ry": "standard",
        "rrup": "ln",
        # "rx": "ln",
        # "rjb": "ln",
        # "ry": "ln",

        "active_shallow": None,
        "volcanic": None,
    },
    "im_config": {
        # "AI": None,
        # "CAV": None,
        # "Ds575": None,
        "Ds595": None,
        # "MMI": None,
        "PGA": None,
        # "PGV": None,
        # "pSA_0.01": None,
        # "pSA_0.02": None,
        # "pSA_0.03": None,
        # "pSA_0.04": None,
        # "pSA_0.05": None,
        # "pSA_0.075": None,
        # "pSA_0.1": None,
        # "pSA_0.12": None,
        # "pSA_0.15": None,
        # "pSA_0.17": None,
        # "pSA_0.2": None,
        # "pSA_0.25": None,
        # "pSA_0.3": None,
        # "pSA_0.4": None,
        "pSA_0.5": None,
        # "pSA_0.6": None,
        # "pSA_0.7": None,
        # "pSA_0.75": None,
        # "pSA_0.8": None,
        # "pSA_0.9": None,
        # "pSA_1.0": None,
        # "pSA_1.25": None,
        # "pSA_1.5": None,
        "pSA_10.0": None,
        # "pSA_2.0": None,
        # "pSA_2.5": None,
        # "pSA_3.0": None,
        # "pSA_4.0": None,
        "pSA_5.0": None,
        # "pSA_6.0": None,
        # "pSA_7.5": None,
    },
}
stats_df = pd.read_csv(
    INPUT_CONFIG["stats_df"], index_col="feature"
)
INPUT_CONFIG["feature_config"] = nn_gmm.convert_input_config(
    INPUT_CONFIG["feature_config"], stats_df
)
INPUT_CONFIG["im_config"] = nn_gmm.convert_input_config(INPUT_CONFIG["im_config"], stats_df)



# Config
CONFIG = {
    "model_config": {
        "hidden_layer_config": {"dropout": 0.3},
        "hidden_layer_func": nn_gmm.relu_dropout,
        # "units": [2048, 2048, 2048],
        # "units": [1024, 1024, 1024, 1024],
        # "units": [512, 512, 512, 512],
        # "units": [256, 256, 256, 256, 256],
        # "units": [32, 32],
        # "units": [128, 128],
        # "units": [128, 128, 128, 128, 128, 128],
        "units": [76, 76, 76],
        # "units": [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, ],
        # "units": [1024, 512, 256, 128]
    },
    "training_config": {
        "batch_size": 1024,
        # "batch_size": 1024,
        "shuffle_buffer_size": int(7e6),
        "n_epochs": 7,
        # "n_epochs": 1,
        "optimizer": "Adam",
        "loss": "mse",
        # "loss": nn_gmm.MargNLLLoss(len(list(INPUT_CONFIG["im_config"].keys()))),
    },
}

if __name__ == "__main__":
    # train_result, output_dir, *_ = nn_gmm.train(
    #     INPUT_CONFIG, CONFIG, model_fn=nn_gmm.create_gaussian_model, verbose=1
    # )

    train_result, output_dir, *_ = nn_gmm.train(
        INPUT_CONFIG, CONFIG, model_fn=nn_gmm.create_reg_model, verbose=1
    )


