import tempfile
from typing import Tuple, Iterable, Callable, Dict, List, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from visualization.gmt.plotting import plot_single, plot_multiple

from . import plotting_utils as plt_utils


MARKERS = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "D",
    "d",
    "|",
    "_",
    "P",
    "X",
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
]


def plot_mag_vs30_res_bins(
    df: pd.DataFrame,
    mean_est_df: pd.DataFrame,
    im: str,
    mag_bins: np.ndarray,
    vs30_bins: np.ndarray,
    output_ffp: Path,
    log_space: bool = False,
):
    mag_ind = np.digitize(df.mag.values, mag_bins)
    vs_30_ind = np.digitize(df.vs30.values, vs30_bins)

    n_rows, n_cols = len(mag_bins) - 1, len(vs30_bins) - 1
    fig = plt_utils.multi_fig((8, 6), n_rows, n_cols)
    outer_grid = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=0.1,
        hspace=0.15,
    )
    subfig_ix = 0
    for cur_mag_bin_ix in range(n_rows):

        # Bin indices start from 1
        cur_mag_bin_ix += 1

        if not np.any(mag_ind == cur_mag_bin_ix):
            continue

        for cur_vs30_bin_ix in range(n_cols):
            cur_vs30_bin_ix += 1
            inner_grid = outer_grid[subfig_ix].subgridspec(
                2,
                2,
                width_ratios=(7, 2),
                height_ratios=(2, 7),
                wspace=0.00,
                hspace=0.00,
            )
            subfig_ix += 1

            ax = fig.add_subplot(inner_grid[1, 0])
            ax_histx = fig.add_subplot(inner_grid[0, 0], sharex=ax)
            ax_histy = fig.add_subplot(inner_grid[1, 1], sharey=ax)

            if not np.any(vs_30_ind == cur_vs30_bin_ix):
                continue

            cur_mask = (mag_ind == cur_mag_bin_ix) & (vs_30_ind == cur_vs30_bin_ix)

            if log_space:
                residual = df.loc[cur_mask, im] - mean_est_df.loc[cur_mask].apply(
                    np.log
                )
                title = f"Mean abs ln residual {np.mean(np.abs(residual.values)):.2f}"
            else:
                residual = (
                    df.loc[cur_mask, im].apply(np.exp) - mean_est_df.loc[cur_mask]
                )
                title = f"Mean abs residual {np.mean(np.abs(residual.values)):.2f}"

            title = (
                f"Vs30: {vs30_bins[cur_vs30_bin_ix - 1]}-{vs30_bins[cur_vs30_bin_ix]}, "
                f"Mag: {mag_bins[cur_mag_bin_ix-1]}-{mag_bins[cur_mag_bin_ix]}, "
                f"N: {df.loc[cur_mask].shape[0]} - {title}"
            )

            residual_scatter_hist_plot(
                df.loc[cur_mask, "rrup"].values,
                residual.values,
                "rrup",
                im,
                fig=fig,
                ax=ax,
                ax_histx=ax_histx,
                ax_histy=ax_histy,
                title=title,
                log_hist_x=True,
                log_hist_y=True,
            )

    fig.savefig(output_ffp)

    plt.close()


def plot_mag_vs30_bins(
    df: pd.DataFrame,
    mean_est_df: pd.Series,
    im: str,
    mag_bins: np.ndarray,
    vs30_bins: np.ndarray,
    output_ffp: Path,
    alpha: float = 0.7,
):
    """Creates a IM vs Rrup scatter plot
    for each of the specified magnitude and vs30 bins
    """
    mag_ind = np.digitize(df.mag.values, mag_bins)
    vs_30_ind = np.digitize(df.vs30.values, vs30_bins)

    n_rows, n_cols = len(mag_bins) - 1, len(vs30_bins) - 1
    fig = plt_utils.multi_fig((16, 10), n_rows, n_cols)
    ax_ix = 1
    for cur_mag_bin_ix in range(n_rows):
        # Bin indices start from 1
        cur_mag_bin_ix += 1

        if not np.any(mag_ind == cur_mag_bin_ix):
            continue

        for cur_vs30_bin_ix in range(n_cols):
            cur_vs30_bin_ix += 1
            cur_ax = fig.add_subplot(n_rows, n_cols, ax_ix)
            ax_ix += 1

            if not np.any(vs_30_ind == cur_vs30_bin_ix):
                continue

            cur_mask = (mag_ind == cur_mag_bin_ix) & (vs_30_ind == cur_vs30_bin_ix)

            cur_ax.scatter(
                df.rrup[cur_mask],
                np.exp(df.loc[cur_mask, im]),
                s=1.0,
                c="royalblue",
                alpha=alpha,
            )
            cur_ax.scatter(
                df.rrup[cur_mask],
                np.exp(mean_est_df.loc[cur_mask].values),
                s=1.0,
                c="darkorange",
                alpha=alpha,
            )
            cur_ax.set_yscale("log")

            cur_ax.text(
                0.99,
                0.99,
                f"Vs30: {vs30_bins[cur_vs30_bin_ix - 1]}-{vs30_bins[cur_vs30_bin_ix]}, "
                f"Mag: {mag_bins[cur_mag_bin_ix-1]}-{mag_bins[cur_mag_bin_ix]},"
                f"N: {df.loc[cur_mask].shape[0]}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=cur_ax.transAxes,
            )
            cur_ax.grid(linestyle="--", linewidth=0.25, alpha=0.75)

    fig.tight_layout()
    fig.savefig(output_ffp)

    plt.close()


# def plot_n_records_map(
#     data_dirs: List[Union[Path, str]],
#     plot_items_ffp: Union[Path, str],
#     output_ffp: str,
#     title: str = "Number-of-records",
# ):
#     """Generates a spatial map that shows number of records at each station"""
#     data_dirs, plot_items_ffp = utils.to_path(data_dirs), utils.to_path(plot_items_ffp)
#
#     ds = data.load_dataset(
#         data_dirs,
#         data.load_feature_details(data_dirs[0]),
#         5_000_000,
#         shuffle_buffer=None,
#         block_size=1024,
#     )
#
#     dfs = []
#     for cur_batch in ds.as_numpy_iterator():
#         cur_df = pd.DataFrame.from_dict(
#             {key: cur_batch[key] for key in ["id", "lat", "lon"]}
#         )
#         cur_df["id"] = cur_df.id.str.decode("UTF-8")
#         cur_df.set_index("id", inplace=True)
#
#         dfs.append(cur_df)
#
#     df = pd.concat(dfs)
#     df["station"] = utils.get_station_from_id(df.index.values.astype(str))
#
#     station_lookup_df = utils.get_station_lookup(df)
#
#     n_records_df = df.groupby("station").count()
#     n_records_df["count"] = n_records_df["lat"]
#     n_records_df.drop(columns=["lat", "lon"], inplace=True)
#
#     n_records_df = pd.merge(
#         n_records_df, station_lookup_df, how="inner", right_index=True, left_index=True
#     )
#
#     cb_options = plt_utils.compute_GMT_std_ticks(n_records_df["count"], non_negative=False)
#     gmt_options = plt_utils.get_gmt_options_dict(
#         options={**{"title": title, "xyz-cpt-labels": "n_records"}, **cb_options,}
#     )
#     csv_ffp = plt_utils.gmt_save(n_records_df, "count", output_ffp, gmt_options=gmt_options)
#
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         plot_single(plot_items_ffp, csv_ffp, plt_utils.DEFAULT_STANDARD_GMT_PLOT_OPTIONS, tmp_dir)
#
#     return


def add_emp(emp_df: pd.DataFrame, label: str = "Bradley 2013", ax: plt.Axes = None):
    """Adds the empirical data to the
    current plot

    Parameters
    ----------
    emp_df: dataframe
        The empirical GMM estimates, expects
        the columns [mu, sigma]
    """
    if ax is None:
        ax = plt.gca()

    # mean prediction
    ax.plot(emp_df.index.values, emp_df["mu"].values, c="r", label=f"{label}", linewidth=0.75)
    # +- sigma
    ax.plot(
        emp_df.index.values,
        emp_df["mu"].values * np.exp(emp_df["sigma"].values),
        c="r",
        linestyle="--",
        # label=f"Std {label}",
        linewidth=0.75
    )
    ax.plot(
        emp_df.index.values,
        emp_df["mu"].values * np.exp(-emp_df["sigma"].values),
        c="r",
        linestyle="--",
        linewidth=0.75
    )


def scatter_hist(
    x, y, ax, ax_histx, ax_histy, n_bins: int = 25, scatter_kwargs: Dict = None
):
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, **scatter_kwargs)

    # bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=n_bins, color="k")
    ax_histy.hist(y, bins=n_bins, orientation="horizontal", color="k")


def residual_scatter_hist_plot(
    x: np.ndarray,
    residual: np.ndarray,
    x_label: str,
    y_label: str,
    title: str = "",
    output_ffp: str = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    ax_histx: plt.Axes = None,
    ax_histy: plt.Axes = None,
    log_hist_y: bool = False,
    log_hist_x: bool = False,
):
    """Creates a scatter + hist (x/y axis) plot"""
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(7, 2),
        height_ratios=(2, 7),
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=0.00,
        hspace=0.00,
    )

    if ax is None:
        fig = plt.figure(figsize=(18, 13.5))
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    scatter_hist(
        x,
        residual,
        ax,
        ax_histx,
        ax_histy,
        n_bins=50,
        scatter_kwargs={"s": 0.5, "alpha": 0.5},
    )
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.axhline(0.0, color="k", linestyle="--")

    under_mask = residual > 0
    over_mask = residual < 0
    ax.text(
        0.01,
        0.99,
        f"Under - N: {np.count_nonzero(under_mask)} - SumE: {np.sum(residual[under_mask]):.2f} - "
        f"SumSE: {np.sum(np.square(residual[under_mask])):.2f}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.01,
        0.01,
        f"Over - N: {np.count_nonzero(over_mask)} - SumE: {np.sum(residual[over_mask]):.2f} - "
        f"SumSE: {np.sum(np.square(residual[over_mask])):.2f}",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    ax_histx.set_title(title)

    if log_hist_x:
        ax_histx.set_yscale("log")
    if log_hist_y:
        ax_histy.set_xscale("log")

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    if output_ffp is not None:
        fig.savefig(output_ffp)
        plt.close()
