from typing import Dict
import os

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

def convert_to_pre_fn(config: Dict, stats_df: pd.DataFrame):
    for key, item in config.items():
        if item is None:
            continue
        elif item == "standard":
            config[key] = nn_gmm.get_standard_scaling_fn(stats_df.loc[key, "mean"], stats_df.loc[key, "std"])
        elif item == "min_max":
            config[key] = nn_gmm.get_min_max_scaling_fn(stats_df.loc[key, "min"], stats_df.loc[key, "max"])
        else:
            raise ValueError(f"{item} is not a valid preprocessing config value")

    return config


INPUT_CONFIG = {
    "train_data_dir": "/nesi/nobackup/nesi00213/nn_gmm/data/sample_files/train",
    "val_data_dir": "/nesi/nobackup/nesi00213/nn_gmm/data/sample_files/val",
    "base_output_dir": "/home/cbs51/code/nn_gmm/results/test",
    "output_dir": None,
    "feature_config": {
        "active_shallow": None,
        "dip": "standard",
        "lat": "min_max",
        "lon": "min_max",
        "mag": "standard",
        "rake": "standard",
        "volcanic": None,
        "vs30": "standard",
        "width": "standard",
        "z1p0": "standard",
        "z2p5": "standard",
        "ztor": "standard",
        "rrup": "standard",
        "rx": "standard",
        "rjb": "standard",
        "ry": "standard",
    },
    "im_config": {
        "AI": "standard",
        "CAV": "standard",
        "Ds575": "standard",
        "Ds595": "standard",
        "MMI": "standard",
        "PGA": "standard",
        "PGV": "standard",
        "pSA_0.01": "standard",
        "pSA_0.02": "standard",
        "pSA_0.03": "standard",
        "pSA_0.04": "standard",
        "pSA_0.05": "standard",
        "pSA_0.075": "standard",
        "pSA_0.1": "standard",
        "pSA_0.12": "standard",
        "pSA_0.15": "standard",
        "pSA_0.17": "standard",
        "pSA_0.2": "standard",
        "pSA_0.25": "standard",
        "pSA_0.3": "standard",
        "pSA_0.4": "standard",
        "pSA_0.5": "standard",
        "pSA_0.6": "standard",
        "pSA_0.7": "standard",
        "pSA_0.75": "standard",
        "pSA_0.8": "standard",
        "pSA_0.9": "standard",
        "pSA_1.0": "standard",
        "pSA_1.25": "standard",
        "pSA_1.5": "standard",
        "pSA_10.0": "standard",
        "pSA_2.0": "standard",
        "pSA_2.5": "standard",
        "pSA_3.0": "standard",
        "pSA_4.0": "standard",
        "pSA_5.0": "standard",
        "pSA_6.0": "standard",
        "pSA_7.5": "standard",
    },
}
stats_df = pd.read_csv(os.path.join(INPUT_CONFIG["train_data_dir"], "stats.csv"), index_col="feature")
INPUT_CONFIG["feature_config"] = convert_to_pre_fn(INPUT_CONFIG["feature_config"], stats_df)
INPUT_CONFIG["im_config"] = convert_to_pre_fn(INPUT_CONFIG["im_config"], stats_df)

# Config
CONFIG = {
    "model_config": {
        "hidden_layer_config": {"dropout": 0.25},
        "hidden_layer_func": nn_gmm.relu_dropout,
        "units": [128, 128],
    },
    "training_config": {
        "batch_size": 1000,
        "n_epochs": 10,
        "optimizer": "Adam",
        "loss": nn_gmm.MargNLLLoss(38),
    },
}

if __name__ == "__main__":
    train_result, output_dir = nn_gmm.train(
        INPUT_CONFIG, CONFIG, model_fn=nn_gmm.create_gaussian_model, verbose=1
    )

    # Plotting doesn't work yet
    exit()

    train_result.save(os.path.join(output_dir, "train_results.pickle"))
    eval_result = nn_gmm.evaluate(train_result, train_data, val_data)
    eval_result.save(os.path.join(output_dir, "eval_results.pickle"))

    # # Residual plots
    # plot_gen = nn_gmm.EvalPlotGen(eval_result)
    # plot_gen.create_comb_res_hist()
    # plot_gen.create_IM_res_hists()
    #
    # # GMT plots
    # plot_items_ffp = "/home/cbs51/code/visualization/visualization/gmt/plot_items.py"
    # plot_gen.create_sigma_maps(plot_items_ffp, n_procs=4)
    # plot_gen.create_nruptures_maps(plot_items_ffp, n_procs=4)
    # plot_gen.create_comb_res_maps(plot_items_ffp, n_procs=4)
    # plot_gen.create_IM_res_maps(plot_items_ffp, n_procs=4)

    # IM vs rrup/mag... plots
    sel_loc_df = nn_gmm.sel_rand_locations(train_data[0], 1000)
    # sel_loc_df = None
    im_plot_gen = nn_gmm.IMvsPlotGen(nn_gmm.GMM.load(train_result.best_model_dir))
    im_plot_gen.gen_plots(
        np.asarray(
            [
                "PGA",
                "PGV",
                "pSA_0.02",
                "pSA_0.1",
                "pSA_0.5",
                "pSA_2.0",
                "pSA_5.0",
                "pSA_10.0",
            ]
        ),
        {"rrup": np.arange(10, 210, 10), "mag": np.arange(4, 8.1, 0.1)},
        output_dir / "visualisation" / "im_plots",
        locations=sel_loc_df,
    ),
    sel_loc_df.to_csv(output_dir / "visualisation" / "im_plots" / "locations.csv")
