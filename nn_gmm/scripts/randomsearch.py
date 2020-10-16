"""Performs a random search in the
specified parameter space & specified constraints
"""
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tensorflow import keras

import ml_tools
import nn_gmm

## -------- Input Config ------------

INPUT_CONFIG = {
    "train_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/train"],
    "val_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/val"],
    "stats_df": "/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/train/stats.csv",
    "base_output_dir": "/home/claudy/dev/work/data/nn_gmm/results/gridsearch",

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
        "width": "standard",
        "length": "standard",
        "is_point_source": None,
        "lon": "min_max",
        "lat": "min_max",
        "vs30": "standard",
        "s": "standard",
        "theta": "min_max",
        # "z1p0": "standard",
        # "z2p5": "standard",
        "rrup": "standard",
        "rx": "standard",
        "rjb": "standard",
        "ry": "standard",
        "active_shallow": None,
        "volcanic": None,
    },
    "im_config": {
        "Ds595": None,
        "PGA": None,
        "pSA_0.5": None,
        "pSA_10.0": None,
        "pSA_5.0": None,
    },
}
stats_df = pd.read_csv(
    INPUT_CONFIG["stats_df"], index_col="feature"
)
INPUT_CONFIG["feature_config"] = nn_gmm.convert_input_config(
    INPUT_CONFIG["feature_config"], stats_df
)
INPUT_CONFIG["im_config"] = nn_gmm.convert_input_config(INPUT_CONFIG["im_config"], stats_df)

# -------- Generic model config -------------

CONFIG = {
    "model_config": {
        "hidden_layer_config": {"dropout": 0.3},
        "hidden_layer_func": nn_gmm.relu_dropout,
        "units": [76, 76, 76],
    },
    "training_config": {
        "batch_size": 1024,
        "shuffle_buffer_size": int(7e6),
        "n_epochs": 2,
        "optimizer": "Adam",
        "loss": "mse",
        # "loss": nn_gmm.MargNLLLoss(len(list(INPUT_CONFIG["im_config"].keys()))),
    },
}

# -------- Search Domain Config ------------

param_config = {
    "learning_rate": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "batch_size": [256, 512, 1024, int(1e4)],
    "n_hidden_layers": stats.randint(1, 6),
    "n_units": [8, 16, 32, 64, 128, 256],
    "dropout": stats.norm(0.3, 0.07),
    "optimizer": [("sgd", 0.2), ("adam", 0.8)]
}

# -------- Other params ------------

n_evals = 25

# ---------- Run ------------
rand_params_gen = ml_tools.RandomParamGenerator(param_config)

output_dir = Path(INPUT_CONFIG["base_output_dir"]) / nn_gmm.create_run_id()
output_dir.mkdir()
INPUT_CONFIG["base_output_dir"] = str(output_dir)

result_dfs = []
for ix in range(n_evals):
    print("---------------------------------------------")
    print(f"Running model number {ix + 1}/{n_evals}")
    # Generate the current model config
    cur_params = next(rand_params_gen.get_param_set())

    # Config creation
    cur_config = CONFIG.copy()
    cur_config["model_config"]["hidden_layer_config"] = {"dropout": cur_params["dropout"]}
    cur_config["training_config"]["batch_size"] = cur_params["batch_size"]

    cur_config["model_config"]["units"] = [cur_params["n_units"] for layer_ix in range(cur_params["n_hidden_layers"])]

    if cur_params["optimizer"] == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=cur_params["learning_rate"])
    elif cur_params["optimizer"] == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=cur_params["learning_rate"])

    cur_config["training_config"]["optimizer"] = optimizer

    # Running the training
    train_result, _, *_ = nn_gmm.train(
        INPUT_CONFIG, cur_config, model_fn=nn_gmm.create_reg_model, verbose=1
    )

    cur_model_id = output_dir.name
    loss_df = pd.read_csv(output_dir / "loss.csv", index_col=0)
    min_val_loss_ix = loss_df["val_loss"].argmin()

    # Saving the results
    cur_result_dict = {}
    cur_result_dict["id"] = cur_model_id
    cur_result_dict["loss"] = loss_df["loss"].iloc[min_val_loss_ix]
    cur_result_dict["val_loss"] = loss_df["val_loss"].iloc[min_val_loss_ix]
    cur_result_dict = {**cur_result_dict, **cur_params}

    cur_result_df = pd.DataFrame.from_dict(cur_result_dict, orient="index").T.set_index("id")
    result_dfs.append(cur_result_df)

    result_df = pd.concat(result_dfs)
    result_df.to_csv(output_dir / "results.csv", index_label="id")




