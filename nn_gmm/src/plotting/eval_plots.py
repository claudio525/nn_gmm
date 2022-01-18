import copy
from pathlib import Path
from typing import Sequence, Dict

import numpy as np
import pandas as pd
from nn_gmm.src import eval
from nn_gmm.src.model import GMM
from nn_gmm.src.plotting.ResPlotGen import ResPlotGen
from nn_gmm.src.plotting.TrendPlotGen import TrendPlotGen
from nn_gmm.src.plotting.BinPlotGen import BinPlotGen
from nn_gmm.src.eval import DEFAULT_CONST_FEATURES
from nn_gmm.src.plotting import plotting_utils
from nn_gmm.src.console import console

from visualization.plot_items_wrapper import plot_multiple


def gen_residual_plots(model_dir: Path, ims: Sequence[str] = None):
    """Generates residual plots for the specified model and IM"""
    # Setup
    fig_output_dir = model_dir / "plots" / "residual_plots"
    fig_output_dir.mkdir(exist_ok=True, parents=True)

    model = GMM.load(model_dir)
    plt_gen = ResPlotGen()

    ims = model.ims if ims is None else ims

    train_db_ffp = model_dir / "train_predictions.hdf5"
    val_db_ffp = model_dir / "val_predictions.hdf5"

    # plt_gen.gen_spectral_bias_plot([train_db_ffp, val_db_ffp], ims, fig_output_dir)
    plt_gen.gen_spectral_bias_std_plot(
        [train_db_ffp, val_db_ffp], ims, fig_output_dir / "spectral_bias_std.png"
    )

    # General residual distribution
    for im in ims:
        plt_gen.gen_res_plot(train_db_ffp, im, fig_output_dir, prefix="train")
        plt_gen.gen_res_plot(val_db_ffp, im, fig_output_dir, prefix="val")

        # Magnitude
        plt_gen.gen_binned_res_plot(
            train_db_ffp,
            im,
            "mag",
            np.asarray([3, 4, 5, 6, 7, 8, 9]),
            3,
            2,
            fig_output_dir,
            prefix="train",
        )
        plt_gen.gen_binned_res_plot(
            val_db_ffp,
            im,
            "mag",
            np.asarray([3, 4, 5, 6, 7, 8, 9]),
            3,
            2,
            fig_output_dir,
            prefix="val",
        )

        # Rrup
        plt_gen.gen_binned_res_plot(
            train_db_ffp,
            im,
            "rrup",
            np.linspace(0, 200, 11),
            5,
            2,
            fig_output_dir,
            prefix="train",
        )
        plt_gen.gen_binned_res_plot(
            val_db_ffp,
            im,
            "rrup",
            np.linspace(0, 200, 11),
            5,
            2,
            fig_output_dir,
            prefix="val",
        )


def gen_rrup_trend_plots(
    model_dir: Path, ims: Sequence[str], const_features: Dict = DEFAULT_CONST_FEATURES
):
    """Creates a Rrup trend plots for the specified IMs"""
    fig_output_dir = model_dir / "plots" / "trend_plots"

    gmm = GMM.load(model_dir)
    tplot = TrendPlotGen(gmm, "rrup", np.linspace(20, 200, 1000))
    for im in ims:
        tplot.gen_trend_plot(const_features, im, fig_output_dir, model_dir=model_dir)


def gen_rrup_bin_plots(model_dir: Path, ims: Sequence[str]):
    """Creates a Rrup based plot for the specified IMs"""
    output_dir = model_dir / "plots" / "bin_plots"
    val_result_db_ffp = model_dir / f"val_predictions.hdf5"
    train_result_db_ffp = model_dir / f"train_predictions.hdf5"

    bin_plot_gen = BinPlotGen()
    for im in ims:
        bin_plot_gen.create_IM_bin_plot(
            im, output_dir, train_result_db_ffp, feature="rrup", prefix="train"
        )
        bin_plot_gen.create_IM_bin_plot(
            im, output_dir, val_result_db_ffp, feature="rrup", prefix="val"
        )


def gen_spatial_bias_std_plots(
    model_dir: Path,
    ims: Sequence[str],
    plot_items_ffp: Path,
    stations_ffp: Path = None,
    n_procs: int = 4,
):
    out_dir = model_dir / "plots" / "spatial_agg"
    out_dir.mkdir(parents=True, exist_ok=True)

    if stations_ffp is None:
        raise NotImplementedError()
    else:
        station_df = pd.read_csv(
            stations_ffp,
            index_col=2,
            delimiter="\s+",
            header=None,
            names=["lon", "lat"],
        )

    # Load the spatial metrics data
    train_spatial_metrics, val_spatial_metrics = eval.load_spatial_metrics(model_dir)

    for cur_metric in train_spatial_metrics.keys():
        console.print(f"Preparing data for metric {cur_metric}")

        gmt_plot_options = plotting_utils.PLOT_TYPE_OPTIONS_MAPPING[cur_metric]

        csv_ffps = []
        for cur_im in ims:
            cur_gmt_plot_options = {"flags": [],
                "options": {
                **plotting_utils.DEFAULT_GMT_CB_OPTIONS[cur_metric],
                **{"title": f"{cur_metric}-{cur_im}",
                   "xyz-cpt-labels": f"{cur_im}",
                   "dpi": 600}},
            }

            # Train
            cur_train_ffp = out_dir / f"train_{cur_metric}_{cur_im.replace('.', 'p')}"
            cur_train_df = pd.merge(
                train_spatial_metrics[cur_metric][cur_im],
                station_df,
                left_index=True,
                right_index=True,
                how="inner",
            )
            cur_train_df.dropna(inplace=True)
            csv_ffps.append(
                plotting_utils.gmt_save(cur_train_df, cur_im, str(cur_train_ffp), cur_gmt_plot_options)
            )
            console.print(
                f"Train: {cur_metric} - {cur_im}, N = {cur_train_df.shape[0]}"
            )

            # Val
            cur_val_ffp = out_dir / f"val_{cur_metric}_{cur_im.replace('.', 'p')}"
            cur_val_df = pd.merge(
                val_spatial_metrics[cur_metric][cur_im],
                station_df,
                left_index=True,
                right_index=True,
                how="inner",
            )
            cur_val_df.dropna(inplace=True)
            csv_ffps.append(
                plotting_utils.gmt_save(cur_val_df, cur_im, str(cur_val_ffp))
            )
            console.print(f"Val: {cur_metric} - {cur_im}, N = {cur_val_df.shape[0]}")

        console.print("Plotting")
        plot_multiple(
            str(plot_items_ffp), gmt_plot_options, in_ffps=csv_ffps, n_procs=n_procs, timeout=180
        )
    print("wtf")
