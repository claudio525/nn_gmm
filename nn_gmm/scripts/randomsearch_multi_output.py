"""Performs a random search in the
specified parameter space & specified constraints
"""
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
from scipy import stats
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

import ml_tools
import nn_gmm

import matplotlib
matplotlib.use("Agg")


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
    # "/home/claudy/dev/work/data/nn_gmm/input_data/site_data/basin_stations"
    "/mnt/win/image/clus/ml_data/nn_gmm/basin_stations"
)

## -------- Input Config ------------

IO_CONFIG = {
    # "train_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/train"],
    # "val_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/val"],
    # "stats_df": "/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/train/stats.csv",
    # # "base_output_dir": "/home/claudy/dev/work/data/nn_gmm/results/gridsearch",
    # "base_output_dir": "/home/claudy/dev/work/data/nn_gmm/results/tmp",

    "train_data_dirs": ["/mnt/win/image/clus/ml_data/nn_gmm/sample_files/train"],
    "val_data_dirs": ["/mnt/win/image/clus/ml_data/nn_gmm/sample_files/val"],
    "stats_df": "/mnt/win/image/clus/ml_data/nn_gmm/sample_files/train/stats.csv",
    "base_output_dir": "/home/cbs51/code/nn_gmm/results/gridsearch",

    "output_dir": None,
    "multi_input": False,
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
        # "width": "standard",
        # "length": "standard",
        "is_point_source": None,

        "lon": "min_max",
        "lat": "min_max",

        "s": "standard",
        "theta": "min_max",

        "vs30": "standard",
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
    "im_config": {
        "Ds595": "standard",
        "PGA": "standard",
        "pSA_0.5": "standard",
        "pSA_10.0": "standard",
        "pSA_5.0": "standard",
    },
}
stats_df = pd.read_csv(
    IO_CONFIG["stats_df"], index_col="feature"
)
IO_CONFIG["feature_config"] = nn_gmm.convert_pre_config(
    IO_CONFIG["feature_config"], stats_df
)
IO_CONFIG["im_config"] = nn_gmm.convert_pre_config(IO_CONFIG["im_config"], stats_df)

# -------- Generic model config -------------

CONFIG = {
    "model_config": {
        "hidden_layer_config": {},
        "hidden_layer_func": nn_gmm.relu_dropout,
    },
    "training_config": {
        "shuffle_buffer_size": int(7e6),
        "n_epochs": 15,
        "use_sample_weights": False,
        "loss": "mse",
        "callbacks": [
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-6,
                min_delta=0.001,
            )
        ]
    },
}

# -------- Search Domain Config ------------

param_config = {
    "learning_rate": [1e-3],
    "batch_size": [128, 256, 512, 1024, int(1e4)],
    "n_core_layers": stats.randint(1, 7),
    "n_head_layers": stats.randint(1, 4),
    "units_per_layer": [16, 32, 64, 128, 256, 512],
    "hydra_units_per_layer": [16, 32, 64, 128, 256],
    "dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "optimizer": ["adam"]
}


# -------- Other params ------------

n_evals = 50

# ---------- Run ------------
use_wandb = True
rand_params_gen = ml_tools.RandomParamGenerator(param_config)

search_id = nn_gmm.create_run_id()
output_dir = Path(IO_CONFIG["base_output_dir"]) / search_id
output_dir.mkdir()
IO_CONFIG["base_output_dir"] = str(output_dir)

# Save the configs
with open(output_dir / "input_config.json", "w") as f:
    json.dump(IO_CONFIG, f, cls=nn_gmm.utils.GenericObjJSONEncoder, indent=4)

with open(output_dir / "base_config.json", "w") as f:
    json.dump(CONFIG, f, cls=nn_gmm.utils.GenericObjJSONEncoder, indent=4)

with open(output_dir / "param_eval_config.json", "w") as f:
    json.dump(param_config, f, cls=nn_gmm.utils.GenericObjJSONEncoder, indent=4)

with open(output_dir / "metadata.json", "w") as f:
    json.dump({"commit_hash": nn_gmm.utils.get_repo_version()}, f, indent=4)

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

    cur_config["model_config"]["units"] = [cur_params["units_per_layer"] for layer_ix in range(cur_params["n_core_layers"])]
    cur_config["model_config"]["output_units"] = [cur_params["hydra_units_per_layer"] for layer_ix in range(cur_params["n_head_layers"])]

    if cur_params["optimizer"] == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=cur_params["learning_rate"])
    elif cur_params["optimizer"] == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=cur_params["learning_rate"])

    cur_config["training_config"]["optimizer"] = optimizer

    cur_io_config = IO_CONFIG.copy()

    if use_wandb:
        run_id = nn_gmm.create_run_id()
        cur_io_config["run_id"] = run_id

        loc_tag = (
            "location-dependence"
            if "lat" in cur_io_config["feature_config"].keys()
            else "location-independence"
        )

        wandb_run = wandb.init(
            project="nn-gmm",
            entity="cbs51",
            tags=["multi_output", loc_tag, "random_search", search_id],
            name=run_id,
        )

        wandb.config.input_config = cur_io_config
        wandb.config.model_config = CONFIG["model_config"]
        wandb.config.training_config = CONFIG["training_config"]
        cur_config["training_config"]["callbacks"].append(WandbCallback())

    # Running the training
    train_result, cur_output_dir, *_ = nn_gmm.train(
        cur_io_config, cur_config["training_config"], model_fn=nn_gmm.create_reg_multi_output_model,
        model_config=cur_config["model_config"], multi_output=True, verbose=2
    )

    cur_model_id = cur_output_dir.name
    loss_df = pd.read_csv(cur_output_dir / "loss.csv", index_col=0)
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

    # Location specific eval
    model = nn_gmm.GMM.load(train_result.best_model_dir)
    loc_vis_dir = cur_output_dir / "visualisation" / "location"
    loc_vis_dir.mkdir(parents=True)
    basin_mean_abs_ln_res = nn_gmm.run_location_eval(
        model, Path(IO_CONFIG["val_data_dirs"][0]), BASIN_DIR, loc_vis_dir, suffix="val"
    )

    if use_wandb:
        # Upload the Basin residuals
        for cur_basin, cur_series in basin_mean_abs_ln_res.items():
            for cur_im, cur_value in cur_series.iteritems():
                wandb.run.summary[
                    f"mean_abs_ln_res_{cur_basin}_{cur_im.replace('.', 'p')}"
                ] = cur_value

        wandb.save(str(cur_output_dir / "input_config.json"))
        wandb.save(str(cur_output_dir / "model_config.json"))
        wandb.save(str(cur_output_dir / "train_config.json"))

        wandb_run.finish()







