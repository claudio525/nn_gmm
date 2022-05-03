import multiprocessing as mp
from pathlib import Path
from typing import Sequence, Dict, Tuple
from importlib import reload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nn_gmm.src import eval
from nn_gmm.src.model import GMM
from nn_gmm.src.plotting.ResPlotGen import ResPlotGen
from nn_gmm.src.plotting.TrendPlotGen import TrendPlotGen
from nn_gmm.src.plotting.BinPlotGen import BinPlotGen
from nn_gmm.src.eval import DEFAULT_CONST_FEATURES, DEFAULT_METRICS
from nn_gmm.src.plotting import plotting_utils
from nn_gmm.src.console import console
from nn_gmm.src.plotting import spatial_plotting
from nn_gmm.src.ResultDB import ResultDB
from nn_gmm.src import data
from nn_gmm.src import utils


def gen_basin_comp_mag_rrup_plots(
    model_dirs: Sequence[Path],
    im: str,
    metrics: Sequence[str],
    regions: Sequence[str],
    basin_dir: Path,
    output_dir: Path,
    model_names: Sequence[str] = None,
    val: bool = False,
):
    """Creates a figure for each metric-region pair,
    showing the metrics trend (wrt. Magnitude and Rrup)
    for the region"""
    console.print(
        f"[orange]This function assumes that all models were trained/validated on the same data. "
        f"Otherwise this plot is not valid.[/]"
    )

    YAXIS_LIMITS = dict(bias=(-1.0, 1.0), sigma=(0.0, 0.8))
    MAG_MIN, MAG_MAX = 3.5, 8.1
    RRUP_MIN, RRUP_MAX = 0, 390

    # Compute the bins
    # As max/min values are the same for all models, just use the first one
    cur_model_dir = model_dirs[0]
    data_df = ResultDB.get_data_static(
        cur_model_dir / "val_predictions.hdf5"
        if val
        else cur_model_dir / "train_predictions.hdf5",
        ["mag", "rrup", "site", "fault", "rupture"],
    )

    # Create the magnitude and rrup bin sizes
    mag_step = 0.5
    mag_bins = np.arange(
        MAG_MIN - (mag_step / 2.0), MAG_MAX + (mag_step / 2.0), mag_step,
    )

    rrup_step = 20
    rrup_bins = np.arange(RRUP_MIN, RRUP_MAX + rrup_step + (rrup_step / 2.0), rrup_step)

    # Compute the data for each model
    train_metric_results = {}
    val_metric_results = {}

    for cur_model_dir, cur_model_name in zip(model_dirs, model_names):
        console.print(f"Processing model {cur_model_dir.stem}")
        (
            cur_train_basin_metrics,
            cur_val_basin_metrics,
        ) = eval.comp_train_val_basin_metrics(
            cur_model_dir,
            basin_dir,
            print_metrics=False,
            mag_bins=mag_bins,
            rrup_bins=rrup_bins,
            metrics=metrics,
        )

        train_metric_results[cur_model_name] = cur_train_basin_metrics
        val_metric_results[cur_model_name] = cur_val_basin_metrics

    metric_results = val_metric_results if val else train_metric_results

    # Get the basin stations
    basin_dict = data.load_basin_stations(basin_dir)

    model_colors = sns.color_palette("hls", len(model_dirs))
    linewidth = 0.5

    prefix = "val" if val else "train"
    for cur_region in regions:
        # Get the current basin/region stations
        if cur_region == "BaseGrid":
            cur_sites_mask = data.get_base_grid_stations_mask(
                data_df.site.values.astype(str)
            )
        else:
            try:
                cur_sites = basin_dict[cur_region]
            except KeyError:
                console.print(
                    f"[red]Failed to load sites for region {cur_region}. Skipping![/]"
                )
                continue

            cur_sites_mask = utils.pandas_isin(
                data_df.site.values.astype(str), cur_sites
            )

        cur_rup_group = data_df.loc[cur_sites_mask].groupby("rupture").first()

        for cur_metric in metrics:
            cur_mag_metric_keys = [
                f"{cur_metric}_mag_{cur_mag}_{cur_mag + mag_step}"
                for cur_mag in mag_bins
            ]
            cur_rrup_metric_keys = [
                f"{cur_metric}_rrup_{cur_rrup}_{cur_rrup + rrup_step}"
                for cur_rrup in rrup_bins
            ]

            # Create the figure
            fig = plt.figure(figsize=(16, 10))

            ax_dict = fig.subplot_mosaic(
                """
                AB
                CD
                CD
                CD
                """
            )

            # Histogram of data in current basin
            ax_dict["A"].hist(cur_rup_group.mag.values, bins=mag_bins, log=True)
            ax_dict["A"].set_ylabel("Number of Ruptures")

            ax_dict["B"].hist(
                data_df.loc[cur_sites_mask].rrup.values, bins=rrup_bins, log=True
            )
            ax_dict["B"].set_ylabel("Number of Datapoints")

            # Plot the models
            for cur_model_dir, cur_model_name, cur_color in zip(
                model_dirs, model_names, model_colors
            ):
                cur_mag_data = pd.Series(
                    index=mag_bins + (mag_step / 2),
                    data=[
                        metric_results[cur_model_name][cur_region][im].get(
                            cur_key, np.nan
                        )
                        for cur_key in cur_mag_metric_keys
                    ],
                    name=cur_metric,
                )
                cur_rrup_data = pd.Series(
                    index=rrup_bins + (rrup_step / 2),
                    data=[
                        metric_results[cur_model_name][cur_region][im].get(
                            cur_key, np.nan
                        )
                        for cur_key in cur_rrup_metric_keys
                    ],
                    name=cur_metric,
                )

                ax_dict["C"].plot(
                    cur_mag_data.index.values,
                    cur_mag_data.values,
                    marker=".",
                    c=cur_color,
                    linewidth=linewidth,
                    label=cur_model_name,
                )
                ax_dict["D"].plot(
                    cur_rrup_data.index.values,
                    cur_rrup_data.values,
                    marker=".",
                    c=cur_color,
                    linewidth=linewidth,
                    label=cur_model_name,
                )

            # Magnitude plot settings
            ax_dict["C"].set_xlabel(f"Magnitude")
            ax_dict["C"].set_ylabel(eval.FANCY_METRICS[cur_metric])
            ax_dict["C"].grid(linewidth=0.5, alpha=0.5, linestyle="--")
            ax_dict["C"].set_ylim(YAXIS_LIMITS.get(cur_metric, (None, None)))
            ax_dict["D"].legend()

            ax_dict["A"].get_shared_x_axes().join(ax_dict["A"], ax_dict["C"])
            ax_dict["A"].set_xticklabels([])

            # Rrup plot settings
            ax_dict["D"].set_xlabel("$R_{Rup}$")
            ax_dict["D"].set_ylabel(eval.FANCY_METRICS[cur_metric])
            ax_dict["D"].grid(linewidth=0.5, alpha=0.5, linestyle="--")
            ax_dict["D"].set_ylim(YAXIS_LIMITS.get(cur_metric, (None, None)))

            ax_dict["B"].get_shared_x_axes().join(ax_dict["B"], ax_dict["D"])
            ax_dict["B"].set_xticklabels([])

            fig.suptitle(cur_region)
            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            fig.savefig(
                output_dir
                / f"{prefix}_{cur_metric}_{cur_region}_{im.replace('.', 'p')}.png"
            )

            plt.close(fig)


def gen_basin_metric_comp_matrix(
    models: Sequence[Tuple[Path, str]],
    im: str,
    metric: str,
    output_ffp: Path,
    val: bool = False,
):
    """Creates a matrix plot the metric
    for the specified models & available basins

    Parameters
    ----------
    models: sequence of tuples
        Each tuple contains the path to the model directory
        and the model name to be used (or None)
    im: string
    metric: string
        The metric to compare
        Has to be a valid metric in the
        computed basin metrics
    val: bool, optional
        Use validation data
    """

    def _load_metric(model_dir: Path, im: str, metric: str, val: bool = False):
        data_dir = (
            model_dir / "val_basin_metrics"
            if val
            else model_dir / "train_basin_metrics"
        )
        metric_dict = {
            cur_ffp.stem: pd.read_csv(cur_ffp, index_col=0).loc[metric, im]
            for cur_ffp in list(data_dir.glob("*.csv"))
        }

        metric_dict["AllStations"] = pd.read_csv(
            model_dir / "train_metrics.csv"
            if not val
            else model_dir / "val_metrics.csv",
            index_col=0,
        ).loc[metric, im]
        return metric_dict

    CMAP_OPTIONS = dict(
        bias=dict(cmap="seismic_r", vmin=-1.0, vmax=1.0),
        sigma=dict(cmap="hot_r", vmin=0.0, vmax=0.8),
        mae=dict(cmap="hot_r", vmin=0.0, vmax=None),
        mse=dict(cmap="hot_r", vmin=0.0, vmax=None),
    )
    # metric_type = metric.split("_")[0]
    # assert metric_type in CB_LABELS

    metric_results = [
        pd.Series(
            _load_metric(cur_model_dir, im, metric, val=val),
            name=cur_model_dir.stem if cur_model_name is None else cur_model_name,
        )
        for cur_model_dir, cur_model_name in models
    ]

    metrics_df = pd.concat(metric_results, axis=1).T

    fig = plt.figure(figsize=(metrics_df.shape[1], metrics_df.shape[0] + 2))
    ax = fig.add_subplot(1, 1, 1)

    model_labels = (
        [cur_col[:11] for cur_col in metrics_df.index.values]
        if models[0][1] is None
        else metrics_df.index.values
    )

    p = ax.matshow(
        metrics_df,
        cmap=CMAP_OPTIONS[metric]["cmap"],
        vmin=CMAP_OPTIONS[metric]["vmin"],
        vmax=CMAP_OPTIONS[metric]["vmax"],
    )
    plt.colorbar(
        p, location="bottom", pad=0.01, label=eval.FANCY_METRICS[metric],
    )
    ax.set_yticks(np.arange(metrics_df.shape[0]), labels=model_labels)
    ax.set_xticks(np.arange(metrics_df.shape[1]), labels=metrics_df.columns.values)
    ax.text(
        0.1,
        1.2,
        f"{eval.FANCY_METRICS[metric]}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize="x-large",
        weight="bold",
    )

    plt.setp(ax.get_xticklabels(), rotation=25, ha="left", rotation_mode="anchor")

    for i in range(metrics_df.shape[0]):
        for j in range(metrics_df.shape[1]):
            ax.text(
                j,
                i,
                f"{metrics_df.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                color="k",
            )

    fig.tight_layout()
    fig.savefig(output_ffp)


def gen_residual_plots(
    model_dir: Path, ims: Sequence[str] = None, sites: Dict[str, str] = None
):
    """Generates residual plots for the specified model, IM and sites"""
    # Setup
    fig_output_dir = model_dir / "plots" / "residual_plots"
    fig_output_dir.mkdir(exist_ok=True, parents=True)

    sites = eval.DEFAULT_EVAL_SITES if sites is None else sites

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

        # Site residuals
        plt_gen.gen_site_res_plot(
            train_db_ffp,
            im,
            list(sites.values()),
            fig_output_dir,
            prefix="train",
            site_names=list(sites.keys()),
        )
        plt_gen.gen_site_res_plot(
            val_db_ffp,
            im,
            list(sites.values()),
            fig_output_dir,
            prefix="val",
            site_names=list(sites.keys()),
        )


def gen_rrup_trend_plots(
    model_dir: Path,
    ims: Sequence[str],
    source_config: Dict = None,
    site_config: Dict = None,
    site_source_config: Dict = None,
    const_features: Dict = None,
):
    """Creates a Rrup trend plots for the specified IMs"""
    fig_output_dir = model_dir / "plots" / "trend_plots"

    # Read constant features from the given config files
    if const_features is None and source_config is not None:
        assert site_config is not None and site_source_config is not None
        const_features = {**source_config, **site_config, **site_source_config}

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

    console.print(f"Saving {prefix} bias plot for {cur_im}")
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
        "train": {
            "PGA": 160,
            "pSA_0.1": 170,
            "pSA_0.5": 180,
            "pSA_1.0": 200,
            "pSA_3.0": 1300,
            "pSA_5.0": 1600,
            "pSA_10.0": 1000,
        },
        "val": {
            "PGA": 30,
            "pSA_0.1": 30,
            "pSA_0.5": 30,
            "pSA_1.0": 30,
            "pSA_3.0": 160,
            "pSA_5.0": 180,
            "pSA_10.0": 110,
        },
    }

    console.print(f"Generating {prefix} sum squared residual plot for {cur_im}")
    cur_grid = spatial_plotting.create_grid(
        spatial_metrics, cur_im, interp_method="linear"
    )

    # cb_max = float(np.round(np.nanquantile(cur_grid.values, 0.98), -1))
    try:
        cb_max = SUM_SQUARED_RESIDUAL_CB_MAX_LOOKUP[prefix][cur_im]
    except KeyError:
        print(
            f"Sum-Squared-Residual {prefix} - {cur_im} - No CB limit in lookup, skipping!"
        )
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
