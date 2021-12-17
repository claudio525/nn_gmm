"""Class for generating residual plots"""
from pathlib import Path
from typing import Union, Tuple, Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import empirical.util.classdef as classdef
import empirical.util.empirical_factory as emp_factory

import ml_tools
from nn_gmm.src import eval
from nn_gmm.src.model import GMM
from nn_gmm.src.ResultDB import ResultDB
from . import plotting_funcs as plt_funcs
from . import plotting_utils as plt_utils


class ResPlotGen:
    def __init__(self, model: GMM, n_bins: int = 100):
        self.model = model

        self.n_bins = n_bins

    def gen_res_plot(self, db_ffp: Path, im: str, output_dir: Path, prefix: str = None):
        """Creates a residual distribution plot"""
        output_dir.mkdir(exist_ok=True, parents=True)

        columns = [im, f"{im}_est"]
        data_df = ResultDB.get_data_static(db_ffp, columns)

        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
        self.add_res_hist(ax, data_df, im)
        ax.legend()

        prefix = f"{prefix}_" if prefix is not None else ""
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}{im.replace('.', 'p')}_residual_distribution.png")
        plt.close()

    def gen_binned_res_plot(
        self,
        db_ffp: Path,
        im: str,
        feature: str,
        feature_bins: np.ndarray,
        n_rows: int,
        n_cols: int,
        output_dir: Path,
        prefix: str = None,
        figsize: Tuple[int, int] = (16, 10),
    ):
        output_dir.mkdir(exist_ok=True, parents=True)

        # Load the data
        data_df = ResultDB.get_data_static(db_ffp, [im, f"{im}_est", feature])

        fig = plt.figure(figsize=figsize, dpi=200)
        for ix in range(len(feature_bins) - 2):
            cur_ax = fig.add_subplot(n_rows, n_cols, ix + 1)

            cur_bin_min, cur_bin_max = feature_bins[ix], feature_bins[ix + 1]
            cur_mask = (data_df[feature] >= cur_bin_min) & (
                data_df[feature] < cur_bin_max
            )
            self.add_res_hist(cur_ax, data_df.loc[cur_mask], im)
            cur_ax.legend(fontsize="small")

            feature_name = plt_utils.get_feature_name(feature)
            cur_ax.set_title(
                f"{cur_bin_min} <= {feature_name} <= {cur_bin_max}, N={np.count_nonzero(cur_mask)}",
                fontsize="medium",
            )

        prefix = f"{prefix}_" if prefix is not None else ""
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}{im.replace('.', 'p')}_{feature}_residual_distributions.png")
        plt.close(fig)

    def add_res_hist(self, ax: plt.Axes, data_df: pd.DataFrame, im: str):
        """Computes the residual and plots the histogram"""
        # Compute the residual
        residual = data_df[im].values - data_df[f"{im}_est"].values
        mean, std = np.mean(residual), np.std(residual)

        # Ploting
        ax.hist(residual, bins=self.n_bins, density=True)
        ax.axvline(
            mean, label=f"$\mu$: {mean:.2f}", linestyle="-", linewidth=0.75, c="k"
        )
        ax.axvline(
            mean + std,
            label=f"$\sigma$: {std:.2f}",
            linestyle="--",
            linewidth=0.75,
            c="k",
        )
        ax.axvline(mean - std, linestyle="--", linewidth=0.75, c="k")

        ax.text(
            0.02,
            0.25,
            f"Overprediction",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize="small",
        )

        ax.text(
            0.98,
            0.25,
            f"Underprediction",
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize="small",
        )

        ax.grid(linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r"PGA, $\epsilon_{{i, j}}$".format(plt_utils.get_im_name(im)))
        ax.set_xlim(-1.0, 1.0)
