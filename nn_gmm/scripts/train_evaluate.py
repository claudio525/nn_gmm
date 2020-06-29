import os

import tensorflow as tf
import numpy as np

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
    # "sample_db_ffp": "/Users/Clus/code/work/nn_gmm/data/sample_dbs/v18p6.h5",
    # "base_output_dir": "/Users/Clus/code/work/nn_gmm/results/test",
    "sample_db_ffp": "/nesi/nobackup/nesi00213/nn_gmm/data/sample_dbs/v18p6.h5",
    "base_output_dir": "/home/cbs51/code/nn_gmm/results/test",
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
        # "hidden_layer_func": hidden_layers.selu_dropout,
        "hidden_layer_func":  nn_gmm.relu_dropout,
        "units": [60],
    },
    "training_config": {
        "val_size": 0.1,
        "batch_size": 64,
        "n_epochs": 25,
        # "optimizer": tf.keras.optimizers.Adam(learning_rate=1e-5),
        "optimizer": "Adam",
        "loss": "MSE",
    },
}

if __name__ == "__main__":
    train_result, output_dir = nn_gmm.train(INPUT_CONFIG, TRAIN_CONFIG)
    train_result.save(os.path.join(output_dir, "train_results.pickle"))
    eval_result = nn_gmm.evaluate(train_result)
    eval_result.save(os.path.join(output_dir, "eval_results.pickle"))

    # Residual plots
    plot_gen = nn_gmm.EvalPlotGen(eval_result)
    plot_gen.create_comb_res_hist()
    plot_gen.create_IM_res_hists()
    #
    # # GMT plots
    # plot_items_ffp = "/path/to/visualization/visualization/gmt/plot_items.py"
    # plot_gen.create_sigma_maps(plot_items_ffp, n_procs=4)
    # plot_gen.create_nruptures_maps(plot_items_ffp, n_procs=4)
    # plot_gen.create_comb_res_maps(plot_items_ffp, n_procs=4)
    # plot_gen.create_IM_res_maps(plot_items_ffp, n_procs=4)

    # IM vs rrup/mag... plots
    sel_loc_df = nn_gmm.sel_rand_locations(train_result.X, 1000)
    im_plot_gen = nn_gmm.IMvsPlotGen(nn_gmm.GMM.load(train_result.best_model_dir))
    im_plot_gen.gen_plots(np.asarray(["PGA", "PGV", "pSA_0.02", "pSA_0.1", "pSA_0.5", "pSA_2.0", "pSA_5.0", "pSA_10.0"]),
                          {"rrup": np.arange(10, 210, 10), "mag": np.arange(4, 8.1, 0.1)}, sel_loc_df, output_dir / "visualisation" / "im_plots")
    sel_loc_df.to_csv(output_dir / "visualisation" / "im_plots" / "locations.csv")

