import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

import nn_gmm

PLOTTING_CONFIG = {
    "ims": [
        "PGA",
        "PGV",
        "pSA_0.02",
        "pSA_0.1",
        "pSA_0.5",
        "pSA_2.0",
        "pSA_5.0",
        "pSA_7.5",
        "pSA_10.0",
    ],
    "model_dir": None,
    "data_dirs": [
        "/home/cbs51/dev/work/data/nn_gmm/sample_files/cybershake_v20p4/train",
        "/home/cbs51/dev/work/data/nn_gmm/sample_files/validation_v20p5p8/train",
        "/home/cbs51/dev/work/data/nn_gmm/sample_files/validation_v20p6p0/train",
    ],
    "output_dir": None,
    "im_plots_feature_dict": {
        "rrup": np.arange(10, 210, 10),
        "mag": np.arange(4, 8.1, 0.1),
    },
}


def create_plots(config):
    if config["model_dir"] is None or config["output_dir"] is None:
        raise ValueError("No model directory or output directory specified")
    model = nn_gmm.GMM.load(config["model_dir"])

    ims = config["ims"]
    data_dirs = nn_gmm.to_path(config["data_dirs"])
    output_dir = nn_gmm.to_path(config["output_dir"])

    feature_details = nn_gmm.load_feature_details(data_dirs[0])

    # Generic IM vs feature plots
    im_plots_features = config["im_plots_feature_dict"]
    if im_plots_features is not None and len(im_plots_features) > 0:
        # Get locations
        print(f"Getting random locations")
        sel_loc_df = nn_gmm.sel_rand_locations(data_dirs, feature_details, 10000)

        im_plot_gen = nn_gmm.IMvsPlotGen(model)
        im_plot_gen.gen_plots(
            ims,
            im_plots_features,
            output_dir=output_dir / "im_plots",
            locations=sel_loc_df,
        )
        sel_loc_df.to_csv(output_dir / "im_plots" / "locations.csv")


if __name__ == '__main__':
    # User provided arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("train_result_dir", type=str, help="The base output directory from a training run")
        args = parser.parse_args()

        base_dir = Path(args.train_result_dir)
        config = {**PLOTTING_CONFIG, **{"model_dir": base_dir / "best_model", "output_dir": base_dir / "visualisation"}}

        create_plots(config)
    else:
        create_plots(PLOTTING_CONFIG)

