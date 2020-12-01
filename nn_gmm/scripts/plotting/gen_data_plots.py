import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

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


PLOTTING_CONFIG = {
    "gen_hist": True,
    # "hist_features": ["mag", "vs30", "rrup"]
    "hist_data_specs": [
        "mag",
        "vs30",
        "rrup",
        nn_gmm.DataSpec("rrup", np.log, "ln"),
        nn_gmm.DataSpec("PGA", np.exp, None),
        nn_gmm.DataSpec("pSA_5.0", np.exp, None),
        nn_gmm.DataSpec("PGA", None, "ln"),
        nn_gmm.DataSpec("pSA_5.0", None, "ln"),
    ],
    "gen_2d_hist": True,
    "2d_hist_data_specs": [
        ("rrup", "mag"),
        ("rrup", nn_gmm.DataSpec("PGA", np.exp, None)),
        ("mag", nn_gmm.DataSpec("PGA", np.exp, None)),
        ("rrup", nn_gmm.DataSpec("pSA_5.0", np.exp, None)),
        ("mag", nn_gmm.DataSpec("pSA_5.0", np.exp, None)),
        (nn_gmm.DataSpec("PGA", np.exp, None), nn_gmm.DataSpec("pSA_5.0", np.exp, None)),
        ("rrup", nn_gmm.DataSpec("PGA", None, "ln")),
        ("mag", nn_gmm.DataSpec("PGA", None, "ln")),
        ("rrup", nn_gmm.DataSpec("pSA_5.0", None, "ln")),
        ("mag", nn_gmm.DataSpec("pSA_5.0", None, "ln")),
        (nn_gmm.DataSpec("PGA", None, "ln"), nn_gmm.DataSpec("pSA_5.0", None, "ln")),
    ],
    "gen_multi_hist": True,
    "multi_hist_spec": [
        ("rrup", "mag", lambda mag_min, mag_max: np.arange(np.floor(mag_min), np.ceil(mag_max) + 1))
    ]
}


def main(data_dir: Path, output_dir: Path):
    plot_gen = nn_gmm.DataExpPlotGen(
        data_dir, output_dir, data_names=["mag", "rrup", "vs30", "PGA", "pSA_5.0", "sample_weight"]
    )

    if PLOTTING_CONFIG["gen_hist"]:
        plot_gen.gen_hist_plots(PLOTTING_CONFIG["hist_data_specs"])

    if PLOTTING_CONFIG["gen_2d_hist"]:
        for data_spec_x, data_spec_y in PLOTTING_CONFIG["2d_hist_data_specs"]:
            plot_gen.gen_2d_hist_plot(data_spec_x, data_spec_y, n_bins=25)

    if PLOTTING_CONFIG["gen_multi_hist"]:
        for data_spec_hist, data_spec_x, bin_x in PLOTTING_CONFIG["multi_hist_spec"]:
            plot_gen.gen_multi_hist_plot(data_spec_hist, data_spec_x, bin_x, n_bins=25)

    plot_gen.gen_2d_hist_plot("rrup", "mag", n_bins=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", type=str, help="Directory that contains the tfrecord files"
    )
    parser.add_argument("output_dir", type=str, help="Output directory")

    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.output_dir))
