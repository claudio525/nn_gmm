from typing import Dict, Sequence, Union
from pathlib import Path

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
            cur_ax.set_xscale("log")

            cur_ax.set_xlim(5, 200)

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
            cur_ax.grid(linestyle="--", linewidth=0.5, alpha=0.5, which="both")

    fig.tight_layout()
    fig.savefig(output_ffp)

    plt.close()


def add_emp(
    emp_df: pd.DataFrame,
    label: str = "Bradley 2013",
    ax: plt.Axes = None,
    plt_kwargs: Dict = None,
):
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

    plt_kwargs = (
        {**plt_utils.DEFAULT_PLOT_KWARGS, **plt_kwargs}
        if plt_kwargs is not None
        else plt_utils.DEFAULT_PLOT_KWARGS
    )

    # mean prediction
    ax.plot(
        emp_df.index.values,
        emp_df["mu"].values,
        c="r",
        label=f"{label}",
        linewidth=plt_kwargs["linewidth"],
    )
    # +- sigma
    ax.plot(
        emp_df.index.values,
        emp_df["mu"].values * np.exp(emp_df["sigma"].values),
        c="r",
        linestyle="--",
        # label=f"Std {label}",
        linewidth=plt_kwargs["linewidth"],
    )
    ax.plot(
        emp_df.index.values,
        emp_df["mu"].values * np.exp(-emp_df["sigma"].values),
        c="r",
        linestyle="--",
        linewidth=plt_kwargs["linewidth"],
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


def gen_spectral_loss_plot(
    loss_dfs: Sequence[pd.DataFrame],
    run_ids: Sequence[str],
    ims: Sequence[str],
    output_ffp: Path,
    val_only: bool = False,
):
    assert all([cur_im.startswith("pSA") or cur_im == "PGA" for cur_im in ims])

    # Create plot
    fig = plt.figure(figsize=(16, 10), dpi=200)
    loss_ax = fig.add_subplot(1, 1, 1)

    # Setup run colours
    run_colours = {
        cur_run_id: cur_color
        for cur_run_id, cur_color in zip(
            run_ids, sns.color_palette("tab10", n_colors=len(run_ids))
        )
    }

    # Get the periods
    periods = np.asarray(
        [float(im.split("_")[-1]) if im.startswith("pSA") else 0 for im in ims]
    )

    loss_max_values = []
    columns = np.asarray([f"{cur_im}_loss" for cur_im in ims])
    for ix, (cur_loss_df, cur_run_id) in enumerate(zip(loss_dfs, run_ids)):
        # Plot validation line
        loss_ax.plot(
            periods,
            cur_loss_df.loc[:, np.char.add("val_", columns)].min(axis=0).values,
            marker=".",
            linewidth=0.75,
            label=f"{cur_run_id}",
            linestyle="--",
            color=run_colours[cur_run_id],
        )

        # Plot training line
        if not val_only:
            loss_ax.plot(
                periods,
                cur_loss_df.loc[:, columns].min(axis=0).values,
                marker=".",
                linewidth=0.75,
                linestyle="-",
                color=run_colours[cur_run_id],
            )

        # Get current y-limit
        loss_max_values.append(np.max(cur_loss_df[columns].values))

    loss_max_value = np.max(loss_max_values) + 0.025
    loss_ax.set_ylim(0, +loss_max_value)
    loss_ax.set_ylabel(r"Loss")
    loss_ax.set_xlabel("Period, T")
    loss_ax.grid(which="both", linewidth=0.5, alpha=0.5)
    loss_ax.semilogx()
    loss_ax.legend()

    fig.tight_layout()
    fig.savefig(output_ffp)


