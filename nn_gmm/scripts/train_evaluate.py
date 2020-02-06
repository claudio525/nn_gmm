import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from nn_gmm import hidden_layers
from nn_gmm import training, evaluation
from nn_gmm import visualisation as vis

INPUT_CONFIG = {
    "sample_db_ffp": "/Users/Clus/code/work/nn_gmm/data/sample_dbs/v18p6.h5",
    "base_output_dir": "/Users/Clus/code/work/nn_gmm/results/test",
    "output_dir": None,
    "ignore_features": ["rtvz"],
    "categorial_features": ["tect_type"],
    "std_scale_features": [
        "vs30",
        "z1p0",
        "z2p5",
        "dip",
        "rake",
        "width",
        "ztor",
        "mag",
        "rjb",
        "rrup",
        "rx",
        "ry",
    ],
    "min_max_scale_features": ["lat", "lon"],
}

# Config
TRAIN_CONFIG = {
    "model_config": {
        "hidden_layer_config": {"dropout": 0.25},
        "hidden_layer_func": hidden_layers.relu_BN_dropout,
        "units": [60, 60, 60],
    },
    "training_config": {"val_size": 0.1, "batch_size": 32, "n_epochs": 5, "loss": "MSE"},
}

if __name__ == "__main__":
    training_result = training.run(INPUT_CONFIG, TRAIN_CONFIG)
    eval_result = evaluation.evaluate(training_result)
    vis.visualisation(eval_result)
