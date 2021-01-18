#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import yaml
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
    "ims": None,
    "model_dir": None,
    # Input/Output
    "train_data_dirs": [
        "/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/train",
    ],
    # "train_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/small_mod_events_sample_files/train",],
    "val_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/val",],
    # "val_data_dirs": ["/home/cbs51/dev/work/data/nn_gmm/input_data/small_mod_events_sample_files/val",],
    "output_dir": None,
    # Plot type flags
    "gen_bin_plots": True,
    "gen_scatter_res_bin_plots": True,
    "gen_event_plots": True,
    "gen_rel_residual_plots": True,
    "gen_rel_plots": True,

    "gen_agg_plots_train": False,
    "gen_agg_plots_val": False,
    # Event based plots
    "event_plot_types": [],
    "train_rep_events": None,
    "val_rep_events": None,
    # Realisation plots
    "rel_plot_types": ["sim", "est_mean", "res_mean"],
    "train_rel_rep_events": ["AlpineF2K"],
    "val_rel_rep_events": ["HopeConway", "AlpineR"],
    "n_rels": 7,
    # Aggregate based plots
    # "agg_plot_types": ["res_mean"],
    # Other
    "plot_items_ffp": "/home/cbs51/dev/work/code/visualization/visualization/gmt/plot_items.py",
    "n_procs": 8,
}


def create_plots(config):
    if config["model_dir"] is None or config["output_dir"] is None:
        raise ValueError("No model directory or output directory specified")
    model = nn_gmm.GMM.load(config["model_dir"])

    ims = config["ims"]
    train_data_dirs = nn_gmm.utils.to_path(config["train_data_dirs"])
    val_data_dirs = nn_gmm.utils.to_path(config["val_data_dirs"])
    output_dir = nn_gmm.utils.to_path(config["output_dir"])

    feature_details = nn_gmm.load_feature_details(train_data_dirs[0])

    # Generic IM vs feature plots
    im_plots_features = config["im_plots_feature_dict"]
    if config["gen_generic_IM_plots"]:
        print(f"Generating generic IM vs feature plots")
        # Get locations
        sel_loc_df = nn_gmm.sel_rand_locations(val_data_dirs, feature_details, 10000)

        im_plot_gen = nn_gmm.IMvsPlotGen(model)
        im_plot_gen.gen_plots(
            ims,
            im_plots_features,
            output_dir=output_dir / "im_plots",
            locations=sel_loc_df,
        )
        sel_loc_df.to_csv(output_dir / "im_plots" / "locations.csv")

    if config["gen_bin_plots"] or config["gen_scatter_res_bin_plots"]:
        bin_plot_gen = nn_gmm.BinPlotGen(
            config["plot_items_ffp"],
            model,
            {"training": train_data_dirs[0], "validation": val_data_dirs[0]},
            output_dir / "bin_plots",
        )

        if config["gen_bin_plots"]:
            print("Generating bin plots")
            for im in ims:
                # bin_plot_gen.create_bin_plot("training", im)
                bin_plot_gen.create_IM_bin_plot("validation", im)

        if config["gen_scatter_res_bin_plots"]:
            print("Generating bin scatter residual plots")
            for im in ims:
                # bin_plot_gen.create_bin_plot("training", im)
                bin_plot_gen.create_IM_res_scatter_bin_plot(
                    "validation", im, log_space=True
                )
                bin_plot_gen.create_IM_res_scatter_bin_plot(
                    "validation", im, log_space=False
                )

    # Event based plots
    if config["gen_event_plots"]:
        print(f"Generating event based plots")
        print(f"Training events")
        gen_event_plots(
            config["train_rep_events"],
            ims,
            model,
            train_data_dirs,
            output_dir / "event_plots" / "train",
            config,
        )

        print(f"Validation events")
        gen_event_plots(
            config["val_rep_events"],
            ims,
            model,
            val_data_dirs,
            output_dir / "event_plots" / "val",
            config,
        )

    if config["gen_rel_residual_plots"]:
        print("Generating realisation residual plots")
        print("Training data")
        gen_rel_residual_plots(
            ims, model, train_data_dirs, output_dir / "agg" / "train", config
        )
        print("Validation data")
        gen_rel_residual_plots(
            ims, model, val_data_dirs, output_dir / "agg" / "val", config
        )

    if config["gen_rel_plots"]:
        print("Generating realisation based plots")
        print("Training events")
        gen_rel_plots(
            config["train_rel_rep_events"],
            ims,
            model,
            train_data_dirs,
            output_dir / "event_plots" / "train",
            config,
        )
        print("Validation events")
        gen_rel_plots(
            config["val_rel_rep_events"],
            ims,
            model,
            val_data_dirs,
            output_dir / "event_plots" / "val",
            config,
        )

    # Aggregate plots
    if config["gen_agg_plots_train"]:
        print(f"Generating aggregate plots - training data")
        gen_agg_plots(ims, model, train_data_dirs, output_dir / "agg" / "train", config)

    if config["gen_agg_plots_val"]:
        print(f"Generating aggregate plots - validation data")
        gen_agg_plots(ims, model, val_data_dirs, output_dir / "agg" / "val", config)


def gen_rel_residual_plots(ims, model, data_dirs, output_dir, config):
    plot_items_ffp, n_procs = config["plot_items_ffp"], config["n_procs"]
    train_agg_plot_gen = nn_gmm.AggPlotGen(
        plot_items_ffp, model, ims, data_dirs, output_dir
    )

    train_agg_plot_gen.plot_realisation_residuals()


def gen_agg_plots(ims, model, data_dirs, output_dir, config):
    plot_items_ffp, n_procs = config["plot_items_ffp"], config["n_procs"]
    train_agg_plot_gen = nn_gmm.AggPlotGen(
        plot_items_ffp, model, ims, data_dirs, output_dir
    )
    for cur_plot_type in config["agg_plot_types"]:
        train_agg_plot_gen.plot_spatial_agg_maps(cur_plot_type, n_procs=n_procs)


def gen_rel_plots(events, ims, model, data_dirs, output_dir, config):
    plot_gen = nn_gmm.EventPlotGen(
        config["plot_items_ffp"], model, data_dirs, output_dir
    )
    plot_gen.plot_realisation_maps(
        events,
        ims,
        config["rel_plot_types"],
        n_procs=config["n_procs"],
        n_rels=config["n_rels"],
    )


def gen_event_plots(events, ims, model, data_dirs, output_dir, config):
    plot_items_ffp, n_procs = config["plot_items_ffp"], config["n_procs"]
    plot_gen = nn_gmm.EventPlotGen(plot_items_ffp, model, data_dirs, output_dir)

    plot_gen.gen_IM_feature_plots(events, ims, "rrup")
    plot_gen.gen_IM_feature_plots(events, ims, "mag")

    event_plot_types = config["event_plot_types"]
    if "sim" in event_plot_types and "est_mean" in event_plot_types:
        sim_cbs = plot_gen.plot_spatial_events_maps(
            events, ims, data_type="sim", n_procs=n_procs
        )
        plot_gen.plot_spatial_events_maps(
            events,
            ims,
            data_type="est_mean",
            events_cb_options=sim_cbs,
            n_procs=n_procs,
        )

    for cur_plot_type in event_plot_types:
        if cur_plot_type not in ["sim", "est_mean"]:
            plot_gen.plot_spatial_events_maps(
                events, ims, data_type=cur_plot_type, n_procs=n_procs
            )



if __name__ == "__main__":
    # User provided arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "train_result_dirs",
            type=str,
            help="The base output directory from a training run",
            nargs="+",
        )
        parser.add_argument(
            "--config_ffp",
            type=str,
            help="yaml config file that contains plotting constants",
            default=Path(__file__).parent / "script_constants.yaml",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            help="Output dir, mainly for testing",
            default=None,
        )
        args = parser.parse_args()

        for cur_train_result_dir in args.train_result_dirs:
            print(f"Processing - {cur_train_result_dir}")
            base_dir = Path(cur_train_result_dir)
            config_ffp = nn_gmm.to_path(args.config_ffp)
            with config_ffp.open() as f:
                const_config = yaml.safe_load(f)

            config = {
                **PLOTTING_CONFIG,
                **{
                    "model_dir": base_dir / "best_model",
                    "output_dir": base_dir / "visualisation"
                    if args.output_dir is None
                    else args.output_dir,
                    "ims": const_config["ims_plotting"],
                    "train_rep_events": const_config["train_rep_events"],
                    "val_rep_events": const_config["val_rep_events"],
                },
            }

            create_plots(config)
    else:
        create_plots(PLOTTING_CONFIG)
