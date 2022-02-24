import multiprocessing as mp
import copy
from pathlib import Path
from typing import Sequence, Dict
from importlib import reload

import numpy as np
import pandas as pd
from nn_gmm.src import eval
from nn_gmm.src.model import GMM
from nn_gmm.src.plotting.ResPlotGen import ResPlotGen
from nn_gmm.src.plotting.TrendPlotGen import TrendPlotGen
from nn_gmm.src.plotting.BinPlotGen import BinPlotGen
from nn_gmm.src.eval import DEFAULT_CONST_FEATURES, DEFAULT_METRICS
from nn_gmm.src.plotting import plotting_utils
from nn_gmm.src.console import console
from nn_gmm.src.plotting import spatial_plotting


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


def gen_spatial_metric_plots(
    model_dir: Path,
    ims: Sequence[str],
    metrics: Sequence[str] = None,
    n_procs: int = 4,
):
    metrics = DEFAULT_METRICS if metrics is None else metrics

    # Output directory
    out_dir = model_dir / "plots" / "spatial_agg"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the spatial metrics data
    train_spatial_metrics, val_spatial_metrics = eval.load_spatial_metrics(model_dir)

    async_results = []
    with mp.Pool(n_procs) as pool:
        for cur_im in ims:
            console.print(f"Processing IM {cur_im}")
            data_columns = ["lon", "lat", cur_im]

            if "bias" in metrics:
                async_results.append(
                    pool.starmap_async(
                        _gen_spatial_bias_plot,
                        [
                            (
                                cur_data["bias"].loc[:, data_columns],
                                cur_im,
                                out_dir,
                                cur_prefix,
                            )
                            for cur_data, cur_prefix in [
                                (train_spatial_metrics, "train"),
                                (val_spatial_metrics, "val"),
                            ]
                        ],
                    )
                )

            if "sigma" in metrics:
                async_results.append(
                    pool.starmap_async(
                        _gen_spatial_sigma_plot,
                        [
                            (
                                cur_data["sigma"].loc[:, data_columns],
                                cur_im,
                                out_dir,
                                cur_prefix,
                            )
                            for cur_data, cur_prefix in [
                                (train_spatial_metrics, "train"),
                                (val_spatial_metrics, "val"),
                            ]
                        ],
                    )
                )

            if "mean_abs_residual" in metrics:
                async_results.append(
                    pool.starmap_async(
                        _gen_mean_abs_residual_plot,
                        [
                            (
                                cur_data["mean_abs_residual"].loc[:, data_columns],
                                cur_im,
                                out_dir,
                                cur_prefix,
                            )
                            for cur_data, cur_prefix in [
                                (train_spatial_metrics, "train"),
                                (val_spatial_metrics, "val"),
                            ]
                        ],
                    )
                )

            if "sum_squared_residual" in metrics:
                async_results.append(
                    pool.starmap_async(
                        _gen_sum_squared_residual_plot,
                        [
                            (
                                cur_data["sum_squared_residual"].loc[:, data_columns],
                                cur_im,
                                out_dir,
                                cur_prefix,
                            )
                            for cur_data, cur_prefix in [
                                (train_spatial_metrics, "train"),
                                (val_spatial_metrics, "val"),
                            ]
                        ],
                    )
                )

            if "count" in metrics:
                async_results.append(
                    pool.starmap_async(
                        _gen_count_plot,
                        [
                            (
                                cur_data["count"].loc[:, data_columns],
                                cur_im,
                                out_dir,
                                cur_prefix,
                            )
                            for cur_data, cur_prefix in [
                                (train_spatial_metrics, "train"),
                                (val_spatial_metrics, "val"),
                            ]
                        ],
                    )
                )

        # Wait for all processes to finish
        for cur_result in async_results:
            cur_result.wait()


def _gen_spatial_bias_plot(
    spatial_metrics: pd.DataFrame, cur_im: str, output_dir: Path, prefix: str
):
    # Have to do this so it works with MP
    # https://github.com/GenericMappingTools/pygmt/issues/217
    import pygmt

    reload(pygmt)

    console.print(f"Generating {prefix} bias plot for {cur_im}")
    cur_grid = spatial_plotting.create_grid(spatial_metrics, cur_im)

    fig = spatial_plotting.gen_region_fig(
        plotting_utils.get_im_name(cur_im)
        + r" - Bias,  <math>\mathbb{E}_{i \in Rup}[\Delta_i]</math>"
    )
    spatial_plotting.plot_grid(
        fig,
        cur_grid,
        "polar",
        (-0.4, 0.4, 0.8 / 16),
        ("darkred", "darkblue"),
        reverse_cmap=True,
    )

    console.print("Saving")
    fig.savefig(
        output_dir / f"{prefix}_{cur_im.replace('.', 'p')}_bias.png",
        dpi=900,
        anti_alias=True,
    )


def _gen_spatial_sigma_plot(
    spatial_metrics: pd.DataFrame, cur_im: str, output_dir: Path, prefix: str
):
    # Have to do this so it works with MP
    # https://github.com/GenericMappingTools/pygmt/issues/217
    import pygmt

    reload(pygmt)

    console.print(f"Generating {prefix} sigma plot for {cur_im}")
    cur_grid = spatial_plotting.create_grid(spatial_metrics, cur_im)

    fig = spatial_plotting.gen_region_fig(
        plotting_utils.get_im_name(cur_im)
        + r" - Standard deviation of Residual, <math>\sigma_{\Delta}</math>"
    )
    spatial_plotting.plot_grid(
        fig,
        cur_grid,
        "hot",
        (0.0, 0.7, 0.7 / 10),
        ("white", "black"),
        reverse_cmap=True,
    )

    fig.savefig(
        output_dir / f"{prefix}_{cur_im.replace('.', 'p')}_sigma.png",
        dpi=900,
        anti_alias=True,
    )


def _gen_mean_abs_residual_plot(
    spatial_metrics: pd.DataFrame, cur_im: str, output_dir: Path, prefix: str
):
    # Have to do this so it works with MP
    # https://github.com/GenericMappingTools/pygmt/issues/217
    import pygmt

    reload(pygmt)

    console.print(f"Generating {prefix} mean absolute residual plot for {cur_im}")
    cur_grid = spatial_plotting.create_grid(spatial_metrics, cur_im)

    fig = spatial_plotting.gen_region_fig(
        plotting_utils.get_im_name(cur_im)
        + " - Mean Absolute Residual, @[\mu_{|\Delta|}@["
    )
    spatial_plotting.plot_grid(
        fig,
        cur_grid,
        "hot",
        (0.0, 0.5, 0.5 / 10),
        ("white", "black"),
        reverse_cmap=True,
    )

    fig.savefig(
        output_dir / f"{prefix}_{cur_im.replace('.', 'p')}_mean_abs_residual.png",
        dpi=900,
        anti_alias=True,
    )


def _gen_sum_squared_residual_plot(
    spatial_metrics: pd.DataFrame, cur_im: str, output_dir: Path, prefix: str
):
    # Have to do this so it works with MP
    # https://github.com/GenericMappingTools/pygmt/issues/217
    import pygmt
    reload(pygmt)

    SUM_SQUARED_RESIDUAL_CB_MAX_LOOKUP = {
        "train": {"PGA": 160,
                  "pSA_0.1": 170,
                  "pSA_0.5": 180,
                  "pSA_1.0": 200,
                  "pSA_3.0": 1300,
                  "pSA_5.0": 1600,
                  "pSA_10.0": 1000,
                  },
        "val": {"PGA": 30,
                "pSA_0.1": 30,
                "pSA_0.5": 30,
                "pSA_1.0": 30,
                "pSA_3.0": 160,
                "pSA_5.0": 180,
                "pSA_10.0": 110,
                }
    }

    console.print(f"Generating {prefix} sum squared residual plot for {cur_im}")
    cur_grid = spatial_plotting.create_grid(spatial_metrics, cur_im, interp_method="linear")

    # cb_max = float(np.round(np.nanquantile(cur_grid.values, 0.98), -1))
    try:
        cb_max = SUM_SQUARED_RESIDUAL_CB_MAX_LOOKUP[prefix][cur_im]
    except KeyError:
        print(f"Sum-Squared-Residual {prefix} - {cur_im} - No CB limit in lookup, skipping!")
        return

    fig = spatial_plotting.gen_region_fig(
        plotting_utils.get_im_name(cur_im)
        + " - Sum Squared Residual, <math>\mathbb{\Sigma}_{i \in Rup}[\Delta_i^2]</math>"
    )
    spatial_plotting.plot_grid(
        fig,
        cur_grid,
        "hot",
        (0.0, cb_max, cb_max / 20),
        # (0, 2000, 2000 / 10),
        ("white", "black"),
        reverse_cmap=True,
    )

    fig.savefig(
        output_dir / f"{prefix}_{cur_im.replace('.', 'p')}_sum_squared_residual.png",
        dpi=900,
        anti_alias=True,
    )

def _gen_count_plot(
    spatial_metrics: pd.DataFrame, cur_im: str, output_dir: Path, prefix: str
):
    # Have to do this so it works with MP
    # https://github.com/GenericMappingTools/pygmt/issues/217
    import pygmt

    reload(pygmt)

    console.print(f"Generating {prefix} count plot for {cur_im}")
    cur_grid = spatial_plotting.create_grid(spatial_metrics, cur_im)

    cb_max = float(np.round(np.nanquantile(cur_grid.values, 0.98), -1))

    fig = spatial_plotting.gen_region_fig("Number of datapoints")
    spatial_plotting.plot_grid(
        fig,
        cur_grid,
        "hot",
        (0, cb_max, cb_max / 20),
        ("white", "black"),
        "Count",
        reverse_cmap=True,
        log_cmap=False,
    )

    fig.savefig(
        output_dir / f"{prefix}_{cur_im.replace('.', 'p')}_count.png",
        dpi=900,
        anti_alias=True,
    )
