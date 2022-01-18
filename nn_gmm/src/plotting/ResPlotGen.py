"""Class for generating residual plots"""
import multiprocessing as mp
from pathlib import Path
from typing import Union, Tuple, Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import empirical.util.classdef as classdef
import empirical.util.empirical_factory as emp_factory

import ml_tools
from nn_gmm.src import eval
from nn_gmm.src.model import GMM
from nn_gmm.src.ResultDB import ResultDB
from . import plotting_funcs as plt_funcs
from . import plotting_utils as plt_utils
from nn_gmm.src.console import console


class ResPlotGen:
    def __init__(self, n_bins: int = 100):
        self.n_bins = n_bins

    def __compute_residuals(
        self, data_df: pd.DataFrame, im: str
    ) -> Tuple[np.ndarray, float, float]:
        """
        Computes the residuals, and mean and standard deviation,
        for the given data
        """
        residual = data_df[im].values - data_df[f"{im}_est"].values
        mean, std = np.mean(residual), np.std(residual)

        return residual, mean, std

    def __compute_residual_stats(
        self, data_df: pd.DataFrame, ims: Sequence[str]
    ) -> pd.DataFrame:
        # Only interested in pSA or PGA
        ims = [im for im in ims if im.startswith("pSA") or im == "PGA"]

        # Get the periods
        periods = np.asarray(
            [float(im.split("_")[-1]) if im.startswith("pSA") else 0 for im in ims]
        )

        # Compute the bias and standard deviation
        bias_values, std_values = [], []
        for cur_im in ims:
            _, cur_bias, cur_std = self.__compute_residuals(data_df, cur_im)
            bias_values.append(cur_bias)
            std_values.append(cur_std)

        # Create dataframe
        res_stats_df = pd.DataFrame(
            data=np.stack((bias_values, std_values, periods), axis=1),
            columns=["bias", "std", "period"],
            index=ims,
        )
        return res_stats_df

    @staticmethod
    def get_colours(ids: Sequence[str]):
        return {
            cur_run_id: cur_color
            for cur_run_id, cur_color in zip(
                np.sort(ids), sns.color_palette("tab10", n_colors=len(ids))
            )
        }

    def gen_spectral_basin_bias_std_plot(
        self,
        output_ffp: Path,
        model_dirs: Sequence[Path],
        basin_station_dir: Path,
        use_train: bool = False,
        use_val: bool = True,
        basin_ids: Sequence[str] = None,
    ):
        """Creates a spectral bias & (residual) standard deviation plot
        for each of the specified models and basins (defaults to all)
        """
        linestyles = ["-", "--"]

        assert (use_val or use_train) and not (
            use_val and use_train
        ), "Either use_train or use_val (default) has to be set (but not both!)"

        # Assume basin metrics have been computed for each model
        basin_metric_data = {
            cur_model_dir.stem: eval.load_basin_metrics(
                cur_model_dir, train_only=use_train, val_only=use_val
            )
            for cur_model_dir in model_dirs
        }

        # console.print("Computing Basin metrics")
        # with mp.Pool(len(model_dirs)) as pool:
        #     results = pool.starmap(
        #         eval.comp_train_val_basin_metrics,
        #         [(cur_dir, basin_station_dir, False, False) for cur_dir in model_dirs],
        #     )
        # basin_metric_data = {
        #     cur_model_dir.stem: cur_result[0] if use_train else cur_result[1]
        #     for cur_result, cur_model_dir in zip(results, model_dirs)
        # }

        ims = GMM.load(model_dirs[0]).ims
        assert all([cur_im.startswith("pSA") or cur_im == "PGA" for cur_im in ims])
        periods = np.asarray(
            [float(im.split("_")[-1]) if im.startswith("pSA") else 0 for im in ims]
        )

        # Setup run colours
        run_ids = list(basin_metric_data.keys())
        basin_ids = (
            basin_ids
            if basin_ids is not None
            else list(basin_metric_data[run_ids[0]].keys())
        )
        basin_colours = self.get_colours(basin_ids)

        # Create plot
        fig = plt.figure(figsize=(16, 10), dpi=200)
        bias_ax = fig.add_subplot(1, 2, 1)
        std_ax = fig.add_subplot(1, 2, 2)

        bias_max, std_max = 0, 0
        for ix, (cur_run_id, cur_metric_data) in enumerate(basin_metric_data.items()):
            for basin_ix, cur_basin_id in enumerate(basin_ids):
                cur_basin_data = basin_metric_data[cur_run_id][cur_basin_id]
                bias_ax.plot(
                    periods,
                    cur_basin_data.loc["bias", ims],
                    marker=".",
                    linewidth=0.75,
                    linestyle=linestyles[ix],
                    c=basin_colours[cur_basin_id],
                    label=cur_basin_id if ix == 0 else None,
                )
                std_ax.plot(
                    periods,
                    cur_basin_data.loc["sigma", ims],
                    marker=".",
                    linewidth=0.75,
                    linestyle=linestyles[ix],
                    c=basin_colours[cur_basin_id],
                    label=cur_run_id if basin_ix == 0 else None,
                )
                bias_max = max(bias_max, cur_basin_data.loc["bias", ims].abs().max())
                std_max = max(std_max, cur_basin_data.loc["sigma", ims].max())

        bias_ax.text(
            0.02,
            0.25,
            f"Overprediction",
            horizontalalignment="left",
            verticalalignment="center",
            transform=bias_ax.transAxes,
            fontsize="small",
        )

        bias_ax.text(
            0.02,
            0.75,
            f"Underprediction",
            horizontalalignment="left",
            verticalalignment="center",
            transform=bias_ax.transAxes,
            fontsize="small",
        )

        bias_ax.set_ylim(-bias_max - 0.025, bias_max + 0.025)
        bias_ax.legend()
        bias_ax.set_ylabel(r"$\mu_{\mathbf{\Delta}}$")
        bias_ax.set_xlabel("Period, T")
        bias_ax.grid(which="both", linewidth=0.5, alpha=0.5)
        bias_ax.semilogx()
        bias_ax.legend()

        std_ax.set_ylim(0, std_max + 0.025)
        std_ax.legend()
        std_ax.set_ylabel(r"$\sigma_{\mathbf{\Delta}}$")
        std_ax.set_xlabel("Period, T")
        std_ax.grid(which="both", linewidth=0.5, alpha=0.5)
        std_ax.semilogx()

        fig.tight_layout()
        fig.savefig(output_ffp)

    def gen_spectral_bias_std_plot(
        self, db_ffps: Sequence[Path], ims: Sequence[str], output_ffp: Path
    ):
        """Creates a figure with two plots:
        - pSA period vs Bias
        - pSA period vs Std
        """
        assert all([cur_im.startswith("pSA") or cur_im == "PGA" for cur_im in ims])

        # Create plot
        fig = plt.figure(figsize=(16, 10), dpi=200)
        bias_ax = fig.add_subplot(1, 2, 1)
        std_ax = fig.add_subplot(1, 2, 2)

        # Setup run colours
        run_ids = np.unique([cur_db_ffp.parent.stem for cur_db_ffp in db_ffps])
        run_colours = {
            cur_run_id: cur_color
            for cur_run_id, cur_color in zip(
                np.sort(run_ids), sns.color_palette("tab10", n_colors=len(run_ids))
            )
        }

        bias_max_values = []
        for ix, cur_db_ffp in enumerate(db_ffps):
            # Get the data
            columns = ims + [f"{im}_est" for im in ims]
            data_df = ResultDB.get_data_static(cur_db_ffp, columns)

            # Compute residual statistics
            res_stats_df = self.__compute_residual_stats(data_df, ims,)

            # Plot line
            cur_suffix = cur_db_ffp.stem.split("_")[0]
            cur_run_id = cur_db_ffp.parent.stem
            bias_ax.plot(
                res_stats_df.period,
                res_stats_df.bias.values,
                marker=".",
                linewidth=0.75,
                label=f"{cur_run_id}" if cur_suffix == "val" else None,
                linestyle="--" if cur_suffix == "val" else None,
                color=run_colours[cur_run_id],
            )
            std_ax.plot(
                res_stats_df.period,
                res_stats_df["std"].values,
                marker=".",
                linewidth=0.75,
                # label=f"{cur_run_id}_{cur_suffix}",
                linestyle="--" if cur_suffix == "val" else None,
                color=run_colours[cur_run_id],
            )

            # Get current y-limit
            bias_max_values.append(np.max(np.abs(res_stats_df.bias.values)))

        bias_max_value = np.max(bias_max_values) + 0.025
        bias_ax.set_ylim(-bias_max_value, +bias_max_value)
        bias_ax.set_ylabel(r"$\mu_{\mathbf{\Delta}}$")
        bias_ax.set_xlabel("Period, T")
        bias_ax.grid(which="both", linewidth=0.5, alpha=0.5)
        bias_ax.semilogx()
        bias_ax.legend()

        std_ax.set_ylabel(r"$\sigma_{\mathbf{\Delta}}$")
        std_ax.set_xlabel("Period, T")
        std_ax.grid(which="both", linewidth=0.5, alpha=0.5)
        std_ax.semilogx()

        fig.tight_layout()

        fig.savefig(output_ffp)

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
        fig.savefig(
            output_dir / f"{prefix}{im.replace('.', 'p')}_residual_distribution.png"
        )
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
        """Creates a binned residual plot for the specified feature"""
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
        fig.savefig(
            output_dir
            / f"{prefix}{im.replace('.', 'p')}_{feature}_residual_distributions.png"
        )
        plt.close(fig)

    def add_res_hist(self, ax: plt.Axes, data_df: pd.DataFrame, im: str):
        """Computes the residual and plots the histogram"""
        # Compute the residuals
        residuals, mean, std = self.__compute_residuals(data_df, im)

        # Ploting
        ax.hist(residuals, bins=self.n_bins, density=True)
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
        ax.set_xlabel(r"{}, $\epsilon_{{i, j}}$".format(plt_utils.get_im_name(im)))
        ax.set_xlim(-1.0, 1.0)


# def gen_spectral_bias_plot(
#     self, db_ffps: Sequence[Path], ims: Sequence[str], output_ffp: Path
# ):
#     """
#     Creates a pSA period vs Bias (+Std) plot for
#     the specified result databases
#     """
#     # Create plot
#     fig = plt.figure(figsize=(16, 10), dpi=200)
#     bias_ax = fig.add_subplot(1, 1, 1)
#
#     # Setup run colours
#     run_ids = np.unique([cur_db_ffp.parent.stem for cur_db_ffp in db_ffps])
#     run_colours = {
#         cur_run_id: cur_color
#         for cur_run_id, cur_color in zip(
#             run_ids, sns.color_palette("tab10", n_colors=len(run_ids))
#         )
#     }
#
#     max_values = []
#     for ix, cur_db_ffp in enumerate(db_ffps):
#         # Get the data
#         columns = ims + [f"{im}_est" for im in ims]
#         data_df = ResultDB.get_data_static(cur_db_ffp, columns)
#
#         # Compute residual statistics
#         cur_suffix = cur_db_ffp.stem.split("_")[0]
#         cur_run_id = cur_db_ffp.parent.stem
#         res_stats_df = self.__compute_residual_stats(
#             data_df,
#             ims,
#         )
#
#         # Plot line
#         bias_ax.errorbar(
#             res_stats_df.period,
#             res_stats_df.bias.values,
#             yerr=res_stats_df["std"].values / 2,
#             marker=".",
#             linewidth=0.75,
#             capsize=7.5,
#             label=f"{cur_run_id}_{cur_suffix}",
#             linestyle="--" if cur_suffix == "val" else None,
#             color=run_colours[cur_run_id],
#         )
#
#         # Get current y-limit
#         max_values.append(
#             np.max(
#                 np.abs(res_stats_df.bias.values) + (res_stats_df["std"].values / 2)
#             )
#             + 0.05
#         )
#
#     max_value = np.max(max_values)
#     bias_ax.set_ylim(-max_value, +max_value)
#     bias_ax.set_ylabel(r"$\mu_{\mathbf{\Delta}}$")
#     bias_ax.set_xlabel("Period, T")
#
#     bias_ax.grid(which="both", linewidth=0.5, alpha=0.5)
#     bias_ax.semilogx()
#     bias_ax.legend()
#     fig.tight_layout()
#
#     fig.savefig(output_ffp)
