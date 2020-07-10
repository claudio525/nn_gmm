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

INPUT_CONFIG = {
    "train_data_dir": "/Users/Clus/code/work/tmp/nn_gmm/data",
    "val_data_dir": "/Users/Clus/code/work/tmp/nn_gmm/data/val_ds",
    # "base_output_dir": "/home/cbs51/code/nn_gmm/results/test",
    "base_output_dir": "/Users/Clus/code/work/nn_gmm/results/test",
    "output_dir": None,
    "feature_config": {
        "active_shallow": None,
        "dip": None,
        "lat": None,
        "lon": None,
        "mag": None,
        "rake": None,
        "volcanic": None,
        "vs30": None,
        "width": None,
        "z1p0": None,
        "z2p5": None,
        "ztor": None,
    },
    "im_config": {
        "AI": None,
        "CAV": None,
        "Ds575": None,
        "Ds595": None,
        "MMI": None,
        "PGA": None,
        "PGV": None,
        "pSA_0.01": None,
        "pSA_0.02": None,
        "pSA_0.03": None,
        "pSA_0.04": None,
        "pSA_0.05": None,
        "pSA_0.075": None,
        "pSA_0.1": None,
        "pSA_0.12": None,
        "pSA_0.15": None,
        "pSA_0.17": None,
        "pSA_0.2": None,
        "pSA_0.25": None,
        "pSA_0.3": None,
        "pSA_0.4": None,
        "pSA_0.5": None,
        "pSA_0.6": None,
        "pSA_0.7": None,
        "pSA_0.75": None,
        "pSA_0.8": None,
        "pSA_0.9": None,
        "pSA_1.0": None,
        "pSA_1.25": None,
        "pSA_1.5": None,
        "pSA_10.0": None,
        "pSA_2.0": None,
        "pSA_2.5": None,
        "pSA_3.0": None,
        "pSA_4.0": None,
        "pSA_5.0": None,
        "pSA_6.0": None,
        "pSA_7.5": None,
    },
}

# Config
TRAIN_CONFIG = {
    "model_config": {
        "hidden_layer_config": {"dropout": 0.25},
        # "hidden_layer_func": hidden_layers.selu_dropout,
        "hidden_layer_func": nn_gmm.relu_dropout,
        "units": [128, 128],
    },
    "training_config": {
        "batch_size": 1024,
        "n_epochs": 1,
        # "optimizer": tf.keras.optimizers.Adam(learning_rate=1e-5),
        "optimizer": "Adam",
        # "loss": "MSE",
        "loss": nn_gmm.MargNLLLoss(38),
    },
}

if __name__ == "__main__":
    train_result, output_dir = nn_gmm.train(
        INPUT_CONFIG, TRAIN_CONFIG, model_fn=nn_gmm.create_gaussian_model, verbose=1
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
