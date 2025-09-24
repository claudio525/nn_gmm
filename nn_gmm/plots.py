import logging
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import ml_tools as mlt

from . import nn_gmm
from . import emp_gmm
from . import analysis
from . import constants
from . import data
from . import plot_utils


logger = logging.getLogger(__name__)


def magnitude_trend_plot(
    result_dir: Path,
    simulation_df: pd.DataFrame,
    fixed_inputs: dict,
    record_limits: dict,
    record_int_ids: np.ndarray,
    device: str,
    cv: bool = False,
    major_line_width: float = 3.0,
    minor_line_width: float = 2.0,
):
    """
    Create magnitude trend plots for different IMs, comparing NN-GMM with empirical GMM predictions.

    Parameters
    ----------
    result_dir : Path
        NN-GMM result directory
    simulation_df : pd.DataFrame
        DataFrame containing simulation IM values
    fixed_inputs : dict
        Dictionary of fixed inputs for the NN-GM and empirical GM model
    record_limits : dict
        Dictionary specifying record selection limits
    record_int_ids : np.ndarray
        Record integer IDs which can be plotted.
        Must be in the simulation DataFrame
    device : str
        Device to run the predictions on (e.g., "cpu" or "cuda")
    """

    assert np.all(
        np.isin(record_int_ids, simulation_df.index)
    ), "record_int_ids must be a subset of the simulation_df index"

    run_config = nn_gmm.RunConfig.from_yaml(result_dir / "run_config.yaml")

    min_mag, max_mag = 5.25, 8.25
    input_df = analysis.get_mag_input_df(
        mag_values=np.linspace(min_mag, max_mag, 250),
        run_config=run_config,
        **fixed_inputs,
    )

    # Get magnitude predictions
    if cv:
        cv_dirs = [d for d in result_dir.glob("cv_*") if d.is_dir()]
        pred_dfs = {
            cur_dir.stem: nn_gmm.run_predictions_dir(cur_dir, input_df, device=device)
            for cur_dir in cv_dirs
        }
        mean_pred_values = np.stack(
            [cur_df[run_config.pred_mean_keys].values for cur_df in pred_dfs.values()]
        )
        mean_pred_da = xr.DataArray(
            data=mean_pred_values,
            dims=["cv", "mag", "im"],
            coords={
                "cv": list(pred_dfs.keys()),
                "mag": input_df["magnitude"].values,
                "im": run_config.pred_mean_keys,
            },
        )
        std_pred_values = np.stack(
            [cur_df[run_config.pred_std_keys].values for cur_df in pred_dfs.values()]
        )
        std_pred_da = xr.DataArray(
            data=std_pred_values,
            dims=["cv", "mag", "im"],
            coords={
                "cv": list(pred_dfs.keys()),
                "mag": input_df["magnitude"].values,
                "im": run_config.pred_std_keys,
            },
        )

    else:
        pred_df = nn_gmm.run_predictions_dir(result_dir, input_df, device=device)
    emp_pred_df = emp_gmm.get_gmm_predictions(input_df, constants.GMM_MAPPING)

    # Get similar records
    similar_record_ids = data.get_similar_records(
        run_config, fixed_inputs, record_limits, record_int_ids=record_int_ids
    )
    logger.info(f"Found {len(similar_record_ids)} similar records")

    # Create magnitude plots for each IM
    plot_ims = ["pSA_0.01", "pSA_0.5", "pSA_1.0", "pSA_5.0"]
    fig, axs = mlt.plotting.get_fig_axes(4, 2, 2, ind_figsize=(8, 6))

    for i, (ax, im) in enumerate(zip(axs, plot_ims)):
        # CV predictions
        # Plot the average mean prediction across all CV folds
        if cv:
            # Plot individual CV predictions
            ax.plot(
                mean_pred_da.coords["mag"].values,
                np.exp(mean_pred_da.sel(im=f"{im}_pred").values.T),
                c="k",
                linestyle="--",
                linewidth=minor_line_width,
            )

            avg_mean = mean_pred_da.mean(dim="cv").sel(im=f"{im}_pred")
            avg_std = std_pred_da.mean(dim="cv").sel(im=f"{im}_pred_std")
            ax.plot(
                mean_pred_da.coords["mag"].values,
                np.exp(avg_mean + avg_std),
                c="b",
                linestyle="--",
                linewidth=minor_line_width,
            )
            ax.plot(
                mean_pred_da.coords["mag"].values,
                np.exp(avg_mean - avg_std),
                c="b",
                linestyle="--",
                linewidth=minor_line_width,
                label="NN-GMM Average Std",
            )
            ax.fill_between(
                mean_pred_da.coords["mag"].values,
                np.exp(avg_mean - avg_std),
                np.exp(avg_mean + avg_std),
                color="b",
                alpha=0.2,
            )

            ax.plot(
                mean_pred_da.coords["mag"].values,
                np.exp(avg_mean),
                c="b",
                label="NN-GMM Average Mean",
                linewidth=major_line_width,
            )
        else:
            ax.plot(
                pred_df["magnitude"],
                np.exp(pred_df[f"{im}_pred"]),
                c="b",
                label="NN-GMM",
                linewidth=major_line_width,
            )
            ax.plot(
                pred_df["magnitude"],
                np.exp(pred_df[f"{im}_pred"] + pred_df[f"{im}_pred_std"]),
                c="b",
                linestyle="--",
                linewidth=minor_line_width,
            )
            ax.plot(
                pred_df["magnitude"],
                np.exp(pred_df[f"{im}_pred"] - pred_df[f"{im}_pred_std"]),
                c="b",
                linestyle="--",
                linewidth=minor_line_width,
            )
            ax.fill_between(
                pred_df["magnitude"],
                np.exp(pred_df[f"{im}_pred"] - pred_df[f"{im}_pred_std"]),
                np.exp(pred_df[f"{im}_pred"] + pred_df[f"{im}_pred_std"]),
                color="b",
                alpha=0.2,
            )

        # Empirical GMM predictions
        ax.plot(
            emp_pred_df["magnitude"],
            np.exp(emp_pred_df[f"{im}_mean"]),
            c="g",
            label="Empirical GMM",
            linewidth=major_line_width,
        )
        ax.plot(
            emp_pred_df["magnitude"],
            np.exp(emp_pred_df[f"{im}_mean"] + emp_pred_df[f"{im}_std_Total"]),
            c="g",
            linestyle="--",
            linewidth=minor_line_width,
        )
        ax.plot(
            emp_pred_df["magnitude"],
            np.exp(emp_pred_df[f"{im}_mean"] - emp_pred_df[f"{im}_std_Total"]),
            c="g",
            linestyle="--",
            linewidth=minor_line_width,
        )
        ax.fill_between(
            emp_pred_df["magnitude"],
            np.exp(emp_pred_df[f"{im}_mean"] - emp_pred_df[f"{im}_std_Total"]),
            np.exp(emp_pred_df[f"{im}_mean"] + emp_pred_df[f"{im}_std_Total"]),
            color="g",
            alpha=0.2,
        )

        ax.scatter(
            simulation_df.loc[similar_record_ids, "magnitude"].values
            + np.random.uniform(-0.01, 0.01, similar_record_ids.size),
            simulation_df.loc[similar_record_ids, im],
            s=1,
            alpha=0.5,
        )

        # ax.set_xlim(5.25, 8.25)
        ax.set_ylabel(im)
        ax.set_xlabel("Magnitude")
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_yscale("log")
        ax.set_xlim(min_mag, max_mag)

        if i == 0:
            ax.legend()

    fig.tight_layout()

    return fig, axs


def rrup_trend_plot(
    result_dir: Path,
    simulation_df: pd.DataFrame,
    fixed_inputs: dict,
    record_limits: dict,
    record_int_ids: np.ndarray,
    device: str,
    cv: bool = False,
    major_line_width: float = 3.0,
    minor_line_width: float = 2.0,
):
    """
    Create rrup trend plots for different IMs, comparing NN-GMM with empirical GMM predictions.

    Parameters
    ----------
    result_dir : Path
        NN-GMM result directory
    simulation_df : pd.DataFrame
        DataFrame containing simulation IM values
    fixed_inputs : dict
        Dictionary of fixed inputs for the NN-GM and empirical GM model
    record_limits : dict
        Dictionary specifying record selection limits
    record_int_ids : np.ndarray
        Record integer IDs which can be plotted
        Must be in the simulation DataFrame
    device : str
        Device to run the predictions on (e.g., "cpu" or "cuda")
    """

    assert np.all(
        np.isin(record_int_ids, simulation_df.index)
    ), "record_int_ids must be a subset of the simulation_df index"

    run_config = nn_gmm.RunConfig.from_yaml(result_dir / "run_config.yaml")

    min_rrup, max_rrup = 0.1, 300
    input_df = analysis.get_rrup_input_df(
        rrup_values=np.linspace(0.1, 300, 250), run_config=run_config, **fixed_inputs
    )

    # Get rrup predictions
    if cv:
        cv_dirs = [d for d in result_dir.glob("cv_*") if d.is_dir()]
        pred_dfs = {
            cur_dir.stem: nn_gmm.run_predictions_dir(cur_dir, input_df, device=device)
            for cur_dir in cv_dirs
        }
        mean_pred_values = np.stack(
            [cur_df[run_config.pred_mean_keys].values for cur_df in pred_dfs.values()]
        )
        mean_pred_da = xr.DataArray(
            data=mean_pred_values,
            dims=["cv", "rrup", "im"],
            coords={
                "cv": list(pred_dfs.keys()),
                "rrup": input_df["rrup"].values,
                "im": run_config.pred_mean_keys,
            },
        )
        std_pred_values = np.stack(
            [cur_df[run_config.pred_std_keys].values for cur_df in pred_dfs.values()]
        )
        std_pred_da = xr.DataArray(
            data=std_pred_values,
            dims=["cv", "rrup", "im"],
            coords={
                "cv": list(pred_dfs.keys()),
                "rrup": input_df["rrup"].values,
                "im": run_config.pred_std_keys,
            },
        )
    else:
        pred_df = nn_gmm.run_predictions_dir(result_dir, input_df, device=device)
    emp_pred_df = emp_gmm.get_gmm_predictions(input_df, constants.GMM_MAPPING)

    # Get similar records
    similar_record_ids = data.get_similar_records(
        run_config, fixed_inputs, record_limits, record_int_ids=record_int_ids
    )
    logger.info(f"Found {len(similar_record_ids)} similar records")

    # Create rrup plots for each IM
    plot_ims = ["pSA_0.01", "pSA_0.5", "pSA_1.0", "pSA_5.0"]
    fig, axs = mlt.plotting.get_fig_axes(4, 2, 2, ind_figsize=(8, 6))

    for i, (ax, im) in enumerate(zip(axs, plot_ims)):
        if cv:
            # Plot individual CV predictions
            ax.plot(
                mean_pred_da.coords["rrup"].values,
                np.exp(mean_pred_da.sel(im=f"{im}_pred").values.T),
                c="k",
                linestyle="--",
                linewidth=minor_line_width,
            )

            avg_mean = mean_pred_da.mean(dim="cv").sel(im=f"{im}_pred")
            avg_std = std_pred_da.mean(dim="cv").sel(im=f"{im}_pred_std")
            ax.plot(
                mean_pred_da.coords["rrup"].values,
                np.exp(avg_mean + avg_std),
                c="b",
                linestyle="--",
                linewidth=minor_line_width,
            )
            ax.plot(
                mean_pred_da.coords["rrup"].values,
                np.exp(avg_mean - avg_std),
                c="b",
                linestyle="--",
                linewidth=minor_line_width,
                label="NN-GMM Average Std",
            )
            ax.fill_between(
                mean_pred_da.coords["rrup"].values,
                np.exp(avg_mean - avg_std),
                np.exp(avg_mean + avg_std),
                color="b",
                alpha=0.2,
            )

            ax.plot(
                mean_pred_da.coords["rrup"].values,
                np.exp(avg_mean),
                c="b",
                label="NN-GMM Average Mean",
                linewidth=major_line_width,
            )
        else:
            ax.plot(
                pred_df["rrup"], np.exp(pred_df[f"{im}_pred"]), c="b", label="NN-GMM"
            )
            ax.plot(
                pred_df["rrup"],
                np.exp(pred_df[f"{im}_pred"] + pred_df[f"{im}_pred_std"]),
                c="b",
                linestyle="--",
                linewidth=1,
            )
            ax.plot(
                pred_df["rrup"],
                np.exp(pred_df[f"{im}_pred"] - pred_df[f"{im}_pred_std"]),
                c="b",
                linestyle="--",
                linewidth=1,
            )
            ax.fill_between(
                pred_df["rrup"],
                np.exp(pred_df[f"{im}_pred"] - pred_df[f"{im}_pred_std"]),
                np.exp(pred_df[f"{im}_pred"] + pred_df[f"{im}_pred_std"]),
                color="b",
                alpha=0.2,
            )

        # Empirical GMM predictions
        ax.plot(
            emp_pred_df["rrup"],
            np.exp(emp_pred_df[f"{im}_mean"]),
            c="g",
            label="Empirical GMM",
        )
        ax.plot(
            emp_pred_df["rrup"],
            np.exp(emp_pred_df[f"{im}_mean"] + emp_pred_df[f"{im}_std_Total"]),
            c="g",
            linestyle="--",
            linewidth=1,
        )
        ax.plot(
            emp_pred_df["rrup"],
            np.exp(emp_pred_df[f"{im}_mean"] - emp_pred_df[f"{im}_std_Total"]),
            c="g",
            linestyle="--",
            linewidth=1,
        )
        ax.fill_between(
            emp_pred_df["rrup"],
            np.exp(emp_pred_df[f"{im}_mean"] - emp_pred_df[f"{im}_std_Total"]),
            np.exp(emp_pred_df[f"{im}_mean"] + emp_pred_df[f"{im}_std_Total"]),
            color="g",
            alpha=0.2,
        )

        ax.scatter(
            simulation_df.loc[similar_record_ids, "rrup"].values,
            simulation_df.loc[similar_record_ids, im],
            s=1,
            alpha=0.5,
        )

        # ax.set_xlim(0.1, 1000)
        ax.set_ylabel(im)
        ax.set_xlabel("$R_{Rup}$ (km)")
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(min_rrup, max_rrup)

        if i == 0:
            ax.legend()

    fig.tight_layout()

    return fig, axs


class BiasStdPlot:

    def __init__(
        self,
        im_set: str = "pSA",
        figsize: tuple = (16, 6),
        bias_ylim: tuple = (-0.8, 0.8),
        std_ylim: tuple = (0, 0.8),
        dpi: int = 100,
    ):
        if im_set == "pSA":
            self.fig, self.ax1, self.ax3 = plot_utils.get_pSA_bias_residual_fig(
                figsize=figsize,
                bias_y_axis_limits=bias_ylim,
                std_y_axis_limits=std_ylim,
                dpi=dpi,
            )
            self.ims = constants.PSA_KEYS
        else:
            raise NotImplementedError()

    def add_results(self, res_df: pd.DataFrame, **plt_kwargs):
        """Adds NN-GMM results to the plot"""
        model_bias, res_std = res_df[self.ims].mean(axis=0), res_df[self.ims].std(
            axis=0
        )

        self.ax1.plot(
            constants.PSA_PERIODS,
            model_bias[constants.PSA_KEYS].values,
            **plt_kwargs,
        )
        self.ax3.plot(
            constants.PSA_PERIODS,
            res_std[constants.PSA_KEYS].values,
            **plt_kwargs,
        )

        return self

    def add_nn_gmm_ind_cv_results(
        self,
        res_df: pd.DataFrame,
        **plt_kwargs,
    ):
        """Adds individual NN-GMM CV results to the plot"""
        cv_bias_df = res_df.groupby("cv_iter", observed=True)[self.ims].mean()

        self.ax1.plot(
            constants.PSA_PERIODS,
            cv_bias_df[constants.PSA_KEYS].T.values,
            **plt_kwargs,
        )

        cv_res_std_df = res_df.groupby("cv_iter", observed=True)[self.ims].std()
        self.ax3.plot(
            constants.PSA_PERIODS,
            cv_res_std_df[constants.PSA_KEYS].T.values,
            **plt_kwargs,
        )

        return self

    def add_nn_gmm_cv_band(self, res_df: pd.DataFrame, **plt_kwargs):
        """Adds NN-GMM CV band to the plot"""
        model_bias, res_std = res_df[self.ims].mean(axis=0), res_df[self.ims].std(
            axis=0
        )

        cv_bias_std = res_df.groupby("cv_iter", observed=True)[self.ims].mean().std(axis=0)
        self.ax1.fill_between(
            constants.PSA_PERIODS,
            model_bias[constants.PSA_KEYS].values
            - cv_bias_std[constants.PSA_KEYS].values,
            model_bias[constants.PSA_KEYS].values
            + cv_bias_std[constants.PSA_KEYS].values,
            **plt_kwargs,
        )

        cv_res_std_std = res_df.groupby("cv_iter", observed=True)[self.ims].std().std(axis=0)
        self.ax3.fill_between(
            constants.PSA_PERIODS,
            res_std[constants.PSA_KEYS].values
            - cv_res_std_std[constants.PSA_KEYS].values,
            res_std[constants.PSA_KEYS].values
            + cv_res_std_std[constants.PSA_KEYS].values,
            **plt_kwargs,
        )

        return self

    # def add_emp_gmm_results(self, emp_res_df: pd.DataFrame, **plt_kwargs):
    #     """Adds empirical GMM results to the plot"""
    #     emp_bias, emp_std = emp_res_df[constants.PSA_KEYS].mean(axis=0), emp_res_df[
    #         constants.PSA_KEYS
    #     ].std(axis=0)

    #     self.ax1.plot(constants.PSA_PERIODS, emp_bias.values, **plt_kwargs)
    #     self.ax3.plot(constants.PSA_PERIODS, emp_std.values, **plt_kwargs)

    #     return self

    def add_legend(self, ax: plt.Axes = None):
        if ax is None:
            self.ax1.legend()
        else:
            ax.legend()
        return self


class GroupedBiasStdPlot(BiasStdPlot):

    def __init__(
        self,
        group_key: str,
        group_bin_edges: list[float],
        group_labels: list[str],
        group_colors: list,
        im_set="pSA",
        figsize=(16, 6),
        bias_ylim=(-0.8, 0.8),
        std_ylim=(0, 0.8),
        dpi=100,
    ):
        super().__init__(im_set, figsize, bias_ylim, std_ylim, dpi)

        self.group_key = group_key
        self.group_bin_key = f"{group_key}_bin"
        self.group_bin_edges = group_bin_edges
        self.group_labels = group_labels
        self.group_colors = group_colors

    def add_grouped_results(
        self, res_df: pd.DataFrame, add_legend_entries: bool, **plt_kwargs
    ):
        """Groups the results and adds to the plot"""
        res_df[self.group_bin_key] = pd.cut(
            res_df[self.group_key],
            bins=self.group_bin_edges,
            labels=self.group_labels,
            include_lowest=True,
        )

        # Bias
        for i, cur_bin_label in enumerate(self.group_labels):
            cur_record_ids = res_df.index[res_df[self.group_bin_key] == cur_bin_label]
            cur_model_bias = res_df.loc[cur_record_ids, constants.PSA_KEYS].mean(axis=0)
            self.ax1.plot(
                constants.PSA_PERIODS,
                cur_model_bias[constants.PSA_KEYS].values,
                c=self.group_colors[i],
                **plt_kwargs,
                label=(
                    f"{cur_bin_label}, N={len(cur_record_ids)}"
                    if add_legend_entries
                    else None
                ),
            )

        # Std
        for i, cur_bin_label in enumerate(self.group_labels):
            cur_record_ids = res_df.index[res_df[self.group_bin_key] == cur_bin_label]
            cur_res_std = res_df.loc[cur_record_ids, constants.PSA_KEYS].std(axis=0)
            self.ax3.plot(
                constants.PSA_PERIODS,
                cur_res_std[constants.PSA_KEYS].values,
                c=self.group_colors[i],
                **plt_kwargs,
            )

        return self

    def add_categorial_results(
        self, res_df: pd.DataFrame, add_legend_entries: bool, **plt_kwargs
    ):
        """Adds categorial results (already grouped) to the plot"""
        # Bias
        for i, cur_bin_label in enumerate(self.group_labels):
            cur_record_ids = res_df.index[res_df[self.group_key] == cur_bin_label]
            cur_model_bias = res_df.loc[cur_record_ids, constants.PSA_KEYS].mean(axis=0)
            self.ax1.plot(
                constants.PSA_PERIODS,
                cur_model_bias[constants.PSA_KEYS].values,
                c=self.group_colors[i],
                **plt_kwargs,
                label=(
                    f"{cur_bin_label}, N={len(cur_record_ids)}"
                    if add_legend_entries
                    else None
                ),
            )

        # Std
        for i, cur_bin_label in enumerate(self.group_labels):
            cur_record_ids = res_df.index[res_df[self.group_key] == cur_bin_label]
            cur_res_std = res_df.loc[cur_record_ids, constants.PSA_KEYS].std(axis=0)
            self.ax3.plot(
                constants.PSA_PERIODS,
                cur_res_std[constants.PSA_KEYS].values,
                c=self.group_colors[i],
                **plt_kwargs,
            )

        return self

    def add_nn_gmm_grouped_cv_band(self, res_df: pd.DataFrame, **plt_kwargs):
        """Adds grouped NN-GMM CV band to the plot"""
        res_df[self.group_bin_key] = pd.cut(
            res_df[self.group_key],
            bins=self.group_bin_edges,
            labels=self.group_labels,
            include_lowest=True,
        )

        # Bias
        for i, cur_bin_label in enumerate(self.group_labels):
            cur_record_ids = res_df.index[res_df[self.group_bin_key] == cur_bin_label]
            cur_model_bias = res_df.loc[cur_record_ids, constants.PSA_KEYS].mean(axis=0)
            cur_cv_bias_std = (
                res_df.loc[cur_record_ids]
                .groupby("cv_iter", observed=True)[constants.PSA_KEYS]
                .mean()
                .std(axis=0)
            )

            self.ax1.fill_between(
                constants.PSA_PERIODS,
                cur_model_bias[constants.PSA_KEYS].values
                - cur_cv_bias_std[constants.PSA_KEYS].values,
                cur_model_bias[constants.PSA_KEYS].values
                + cur_cv_bias_std[constants.PSA_KEYS].values,
                color=self.group_colors[i],
                **plt_kwargs,
            )

        # Std
        for i, cur_bin_label in enumerate(self.group_labels):
            cur_record_ids = res_df.index[res_df[self.group_bin_key] == cur_bin_label]
            cur_res_std = res_df.loc[cur_record_ids, constants.PSA_KEYS].std(axis=0)
            cur_cv_res_std_std = (
                res_df.loc[cur_record_ids]
                .groupby("cv_iter", observed=True)[constants.PSA_KEYS]
                .std()
                .std(axis=0)
            )

            self.ax3.fill_between(
                constants.PSA_PERIODS,
                cur_res_std[constants.PSA_KEYS].values
                - cur_cv_res_std_std[constants.PSA_KEYS].values,
                cur_res_std[constants.PSA_KEYS].values
                + cur_cv_res_std_std[constants.PSA_KEYS].values,
                color=self.group_colors[i],
                **plt_kwargs,
            )

        return self
