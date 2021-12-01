"""Train a simple PGA with standard parameters (i.e. no location dependence)"""
import time
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

feature_config = {
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
}
im_config = {
    "PGA": None,
}
data_columns = list(feature_config.keys()) + list(im_config.keys()) + ["id"]

train_data_ffp = Path("/home/claudy/dev/work/data/nn_gmm/training_data/training_data.hdf5")
val_data_ffp = Path("/home/claudy/dev/work/data/nn_gmm/training_data/validation_data.hdf5")

start_time = time.time()
val_df = nn_gmm.TableDB.get_df(val_data_ffp, "id", data_columns)
print(f"Took {time.time() - start_time}")

start_time = time.time()
train_df = nn_gmm.TableDB.get_df_mp(train_data_ffp, "id", data_columns, n_procs=8)
print(f"Took {time.time() - start_time}")

print("wtf")