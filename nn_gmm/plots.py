import logging
from pathlib import Path

import matplotlib.pyplot as plt
import ml_tools as mlt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xarray as xr

from . import analysis, constants, data, emp_gmm, hazard, nn_gmm, plot_utils, utils

logger = logging.getLogger(__name__)


def magnitude_trend_plot(
    result_dir: Path,
    fixed_inputs: dict,
    record_limits: dict,
    device: str,
    simulation_df: pd.DataFrame | None = None,
    record_int_ids: np.ndarray | None = None,
    cv: bool = False,
    plot_ind_cv: bool = True,
    major_line_width: float = 3.0,
    minor_line_width: float = 2.0,
    ind_fig_size: tuple = (8, 6),
    ims: list[str] | None = None,
    dpi: float | None = None,
    axs: list[plt.Axes] | None = None,
    nn_color: str = "blue",
    plot_empirical: bool = True,
    legend_labels: bool = True,
    legend: bool = True,
    fill_between: bool = True,
    min_mag: float = 5.0,
    max_mag: float = 8.25,
):
    """
    Create magnitude trend plots for different IMs, comparing NN-GMM with empirical GMM predictions.

    Parameters
    ----------
    result_dir : Path
        NN-GMM result directory
    fixed_inputs : dict
        Dictionary of fixed inputs for the NN-GM and empirical GM model
    record_limits : dict
        Dictionary specifying record selection limits
    device : str
        Device to run the predictions on (e.g., "cpu" or "cuda")
    simulation_df : pd.DataFrame | None, optional
        DataFrame containing simulation IM values
    record_int_ids : np.ndarray | None, optional
        Record integer IDs which can be plotted.
        Must be in the simulation DataFrame
    cv: bool, optional
        Whether the the result dir is for CV results.
    """
    assert simulation_df is None or np.all(
        np.isin(record_int_ids, simulation_df.index)
    ), "record_int_ids must be a subset of the simulation_df index"
    assert axs is None or len(axs) == (
        len(ims) if ims is not None else len(constants.PLOT_IMS)
    ), "If axs is provided, its length must match the number of ims to be plotted."

    run_config = nn_gmm.load_config(result_dir / "run_config.yaml")

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

    # Get empirical GMM predictions
    emp_pred_df = emp_gmm.get_gmm_predictions(input_df, constants.GMM_MAPPING)

    # Get similar records
    if simulation_df is not None:
        similar_record_ids = data.get_similar_records(
            run_config, fixed_inputs, record_limits, record_int_ids=record_int_ids
        )
        logger.info(f"Found {len(similar_record_ids)} similar records")

    # Create magnitude plots for each IM
    ims = constants.PLOT_IMS if ims is None else ims
    fig = None
    if axs is None:
        fig, axs = mlt.plotting.get_fig_axes(
            len(ims), 2, -1, ind_figsize=ind_fig_size, dpi=dpi
        )

    for i, (ax, im) in enumerate(zip(axs, ims)):
        if i % 2 == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_ticks_position("right")

        # Add data points
        if simulation_df is not None:
            ax.scatter(
                simulation_df.loc[similar_record_ids, "magnitude"].values
                + np.random.uniform(-0.01, 0.01, similar_record_ids.size),
                simulation_df.loc[similar_record_ids, im],
                s=1,
                alpha=0.5,
                c="gray",
            )

        # Empirical GMM predictions
        if plot_empirical:
            ax.plot(
                emp_pred_df["magnitude"],
                np.exp(emp_pred_df[f"{im}_mean"]),
                c="g",
                label="Empirical GMM" if legend_labels else None,
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
            if fill_between:
                ax.fill_between(
                    emp_pred_df["magnitude"],
                    np.exp(emp_pred_df[f"{im}_mean"] - emp_pred_df[f"{im}_std_Total"]),
                    np.exp(emp_pred_df[f"{im}_mean"] + emp_pred_df[f"{im}_std_Total"]),
                    color="g",
                    alpha=0.2,
                )

        # CV predictions
        # Plot the average mean prediction across all CV folds
        if cv:
            # Plot individual CV predictions
            if plot_ind_cv:
                ax.plot(
                    mean_pred_da.coords["mag"].values,
                    np.exp(mean_pred_da.sel(im=f"{im}_pred").values.T),
                    c="k",
                    linestyle="--",
                    linewidth=minor_line_width,
                )

            comb_mean = mean_pred_da.mean(dim="cv").sel(im=f"{im}_pred")
            within_model_std = std_pred_da.mean(dim="cv").sel(im=f"{im}_pred_std")
            between_model_std = mean_pred_da.std(dim="cv").sel(im=f"{im}_pred")
            comb_std = np.sqrt(within_model_std**2 + between_model_std**2)

            ax.plot(
                mean_pred_da.coords["mag"].values,
                np.exp(comb_mean + comb_std),
                c=nn_color,
                linestyle="--",
                linewidth=minor_line_width,
            )
            ax.plot(
                mean_pred_da.coords["mag"].values,
                np.exp(comb_mean - comb_std),
                c=nn_color,
                linestyle="--",
                linewidth=minor_line_width,
                label="NN-GMM Average Std" if legend_labels else None,
            )
            if fill_between:
                ax.fill_between(
                    mean_pred_da.coords["mag"].values,
                    np.exp(comb_mean - comb_std),
                    np.exp(comb_mean + comb_std),
                    color=nn_color,
                    alpha=0.2,
                )

            ax.plot(
                mean_pred_da.coords["mag"].values,
                np.exp(comb_mean),
                c=nn_color,
                label="NN-GMM Average Mean" if legend_labels else None,
                linewidth=major_line_width,
            )
        else:
            ax.plot(
                pred_df["magnitude"],
                np.exp(pred_df[f"{im}_pred"]),
                c=nn_color,
                label="NN-GMM" if legend_labels else None,
                linewidth=major_line_width,
            )
            ax.plot(
                pred_df["magnitude"],
                np.exp(pred_df[f"{im}_pred"] + pred_df[f"{im}_pred_std"]),
                c=nn_color,
                linestyle="--",
                linewidth=minor_line_width,
            )
            ax.plot(
                pred_df["magnitude"],
                np.exp(pred_df[f"{im}_pred"] - pred_df[f"{im}_pred_std"]),
                c=nn_color,
                linestyle="--",
                linewidth=minor_line_width,
            )
            if fill_between:
                ax.fill_between(
                    pred_df["magnitude"],
                    np.exp(pred_df[f"{im}_pred"] - pred_df[f"{im}_pred_std"]),
                    np.exp(pred_df[f"{im}_pred"] + pred_df[f"{im}_pred_std"]),
                    color=nn_color,
                    alpha=0.2,
                )

        ax.set_ylabel(utils.get_nice_im_name(im))
        ax.set_xlabel("Magnitude")
        ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_yscale("log")
        ax.set_xlim(min_mag, max_mag)

        if i == 0 and legend:
            ax.legend()

    if fig:
        fig.tight_layout()
    return fig, axs


def rrup_trend_plot(
    result_dir: Path,
    fixed_inputs: dict,
    record_limits: dict,
    device: str,
    simulation_df: pd.DataFrame | None = None,
    record_int_ids: np.ndarray | None = None,
    cv: bool = False,
    plot_ind_cv: bool = True,
    major_line_width: float = 3.0,
    minor_line_width: float = 2.0,
    ind_fig_size: tuple = (8, 6),
    ims: list[str] | None = None,
    dpi: float | None = None,
    axs: list[plt.Axes] | None = None,
    nn_color: str = "blue",
    plot_empirical: bool = True,
    legend_labels: bool = True,
    legend: bool = True,
    fill_between: bool = True,
):
    """
    Create rrup trend plots for different IMs, comparing NN-GMM with empirical GMM predictions.

    Parameters
    ----------
    result_dir : Path
        NN-GMM result directory
    fixed_inputs : dict
        Dictionary of fixed inputs for the NN-GM and empirical GM model
    record_limits : dict
        Dictionary specifying record selection limits
    simulation_df : pd.DataFrame | None, optional
        DataFrame containing simulation IM values
    record_int_ids : np.ndarray | None, optional
        Record integer IDs which can be plotted
        Must be in the simulation DataFrame
    device : str
        Device to run the predictions on (e.g., "cpu" or "cuda")
    """
    assert simulation_df is None or np.all(
        np.isin(record_int_ids, simulation_df.index)
    ), "record_int_ids must be a subset of the simulation_df index"
    assert axs is None or len(axs) == (
        len(ims) if ims is not None else len(constants.PLOT_IMS)
    ), "If axs is provided, its length must match the number of ims to be plotted."

    run_config = nn_gmm.load_config(result_dir / "run_config.yaml")

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

    # Get empirical GM predictions
    emp_pred_df = emp_gmm.get_gmm_predictions(input_df, constants.GMM_MAPPING)

    # Get similar records
    similar_record_ids = data.get_similar_records(
        run_config, fixed_inputs, record_limits, record_int_ids=record_int_ids
    )
    logger.info(f"Found {len(similar_record_ids)} similar records")

    # Create rrup plots for each IM
    ims = constants.PLOT_IMS if ims is None else ims
    fig = None
    if axs is None:
        fig, axs = mlt.plotting.get_fig_axes(
            len(ims), 2, -1, ind_figsize=ind_fig_size, dpi=dpi
        )

    for i, (ax, im) in enumerate(zip(axs, ims)):
        if i % 2 == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_ticks_position("right")

        # Add data points
        if simulation_df is not None:
            ax.scatter(
                simulation_df.loc[similar_record_ids, "rrup"].values,
                # + np.random.uniform(-0.01, 0.01, similar_record_ids.size),
                simulation_df.loc[similar_record_ids, im],
                s=1,
                alpha=0.5,
                c="gray",
            )
            # if "basin" in simulation_df.columns:
            #     mask = simulation_df.loc[similar_record_ids].basin != "NiB"
            #     ax.scatter(
            #         simulation_df.loc[similar_record_ids].loc[mask, "rrup"].values,
            #         simulation_df.loc[similar_record_ids].loc[mask, im],
            #         s=1,
            #         alpha=0.5,
            #         c="blue",
            #     )

        # Empirical GMM predictions
        if plot_empirical:
            ax.plot(
                emp_pred_df["rrup"],
                np.exp(emp_pred_df[f"{im}_mean"]),
                c="g",
                label="Empirical GMM" if legend_labels else None,
                linewidth=major_line_width,
            )
            ax.plot(
                emp_pred_df["rrup"],
                np.exp(emp_pred_df[f"{im}_mean"] + emp_pred_df[f"{im}_std_Total"]),
                c="g",
                linestyle="--",
                linewidth=minor_line_width,
            )
            ax.plot(
                emp_pred_df["rrup"],
                np.exp(emp_pred_df[f"{im}_mean"] - emp_pred_df[f"{im}_std_Total"]),
                c="g",
                linestyle="--",
                linewidth=minor_line_width,
            )
            if fill_between:
                ax.fill_between(
                    emp_pred_df["rrup"],
                    np.exp(emp_pred_df[f"{im}_mean"] - emp_pred_df[f"{im}_std_Total"]),
                    np.exp(emp_pred_df[f"{im}_mean"] + emp_pred_df[f"{im}_std_Total"]),
                    color="g",
                    alpha=0.2,
                )

        # CV predictions
        # Plot the average mean prediction across all CV folds
        if cv:
            # Plot individual CV predictions
            if plot_ind_cv:
                ax.plot(
                    mean_pred_da.coords["rrup"].values,
                    np.exp(mean_pred_da.sel(im=f"{im}_pred").values.T),
                    c="k",
                    linestyle="--",
                    linewidth=minor_line_width,
                )

            comb_mean = mean_pred_da.mean(dim="cv").sel(im=f"{im}_pred")
            within_model_std = std_pred_da.mean(dim="cv").sel(im=f"{im}_pred_std")
            between_model_std = mean_pred_da.std(dim="cv").sel(im=f"{im}_pred")
            comb_std = np.sqrt(within_model_std**2 + between_model_std**2)

            ax.plot(
                mean_pred_da.coords["rrup"].values,
                np.exp(comb_mean + comb_std),
                c=nn_color,
                linestyle="--",
                linewidth=minor_line_width,
            )
            ax.plot(
                mean_pred_da.coords["rrup"].values,
                np.exp(comb_mean - comb_std),
                c=nn_color,
                linestyle="--",
                linewidth=minor_line_width,
                label="NN-GMM Average Std" if legend_labels else None,
            )
            if fill_between:
                ax.fill_between(
                    mean_pred_da.coords["rrup"].values,
                    np.exp(comb_mean - comb_std),
                    np.exp(comb_mean + comb_std),
                    color=nn_color,
                    alpha=0.2,
                )
            ax.plot(
                mean_pred_da.coords["rrup"].values,
                np.exp(comb_mean),
                c=nn_color,
                label="NN-GMM Average Mean" if legend_labels else None,
                linewidth=major_line_width,
            )
        else:
            ax.plot(
                pred_df["rrup"],
                np.exp(pred_df[f"{im}_pred"]),
                c=nn_color,
                label="NN-GMM" if legend_labels else None,
                linewidth=major_line_width,
            )
            ax.plot(
                pred_df["rrup"],
                np.exp(pred_df[f"{im}_pred"] + pred_df[f"{im}_pred_std"]),
                c=nn_color,
                linestyle="--",
                linewidth=minor_line_width,
            )
            ax.plot(
                pred_df["rrup"],
                np.exp(pred_df[f"{im}_pred"] - pred_df[f"{im}_pred_std"]),
                c=nn_color,
                linestyle="--",
                linewidth=minor_line_width,
            )
            if fill_between:
                ax.fill_between(
                    pred_df["rrup"],
                    np.exp(pred_df[f"{im}_pred"] - pred_df[f"{im}_pred_std"]),
                    np.exp(pred_df[f"{im}_pred"] + pred_df[f"{im}_pred_std"]),
                    color=nn_color,
                    alpha=0.2,
                )

        ax.set_ylabel(im)
        ax.set_xlabel("Source-to-Site Distance, $R_{Rup}$ (km)")
        ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(min_rrup, max_rrup)

        if i == 0 and legend:
            ax.legend()

    if fig:
        fig.tight_layout()

    return fig, axs


class BiasStdPlot:

    def __init__(
        self,
        pSA_keys_periods: tuple[list[str], list[float]] | None = None,
        im_set: str = "pSA",
        figsize: tuple = (16, 6),
        bias_ylim: tuple[float, float] = (-0.8, 0.8),
        std_ylim: tuple[float, float] = (0, 0.8),
        x_axis_limits: tuple[float, float] = (0.01, 10.0),
        **fig_kwargs,
    ):
        """
        Initializes the BiasStdPlot class.

        Parameters
        ----------
        pSA_keys_periods : tuple[list[str], list[float]], optional
            Tuple containing lists of pSA keys and their corresponding periods.
            If None, im_set is used to determine the pSA keys and periods.
        im_set : str, optional
            The IM set to use. Default is "pSA".
        """
        if pSA_keys_periods is not None:
            self.pSA_keys, self.pSA_periods = pSA_keys_periods
        else:
            if im_set == "pSA":
                self.pSA_keys = constants.PSA_KEYS
                self.pSA_periods = constants.PSA_PERIODS
            else:
                raise NotImplementedError()

        self.fig, self.ax1, self.ax3 = plot_utils.get_pSA_bias_residual_fig(
            figsize=figsize,
            bias_y_axis_limits=bias_ylim,
            std_y_axis_limits=std_ylim,
            x_axis_limits=x_axis_limits,
            **fig_kwargs,
        )

    def add_bias(self, model_bias: pd.Series, **plt_kwargs):
        """Adds bias to the plot"""
        self.ax1.plot(
            model_bias.index,
            model_bias.values,
            **plt_kwargs,
        )
        return self

    def add_std(self, res_std: pd.Series, **plt_kwargs):
        """Adds standard deviation to the plot"""
        self.ax3.plot(
            res_std.index,
            res_std.values,
            **plt_kwargs,
        )
        return self

    def add_results(
        self,
        res_df: pd.DataFrame,
        **plt_kwargs,
    ):
        """Adds NN-GMM results to the plot"""
        model_bias, res_std = res_df[self.pSA_keys].mean(axis=0), res_df[
            self.pSA_keys
        ].std(axis=0)

        self.ax1.plot(
            self.pSA_periods,
            model_bias[self.pSA_keys].values,
            **plt_kwargs,
        )
        self.ax3.plot(
            self.pSA_periods,
            res_std[self.pSA_keys].values,
            **plt_kwargs,
        )

        return self

    def add_nn_gmm_ind_cv_results(
        self,
        res_df: pd.DataFrame,
        **plt_kwargs,
    ):
        """Adds individual NN-GMM CV results to the plot"""
        cv_bias_df = res_df.groupby("cv_iter", observed=True)[self.pSA_keys].mean()

        self.ax1.plot(
            self.pSA_periods,
            cv_bias_df[self.pSA_keys].T.values,
            **plt_kwargs,
        )

        cv_res_std_df = res_df.groupby("cv_iter", observed=True)[self.pSA_keys].std()
        self.ax3.plot(
            self.pSA_periods,
            cv_res_std_df[self.pSA_keys].T.values,
            **plt_kwargs,
        )

        return self

    def add_nn_gmm_cv_band(self, res_df: pd.DataFrame, **plt_kwargs):
        """Adds NN-GMM CV band to the plot"""
        model_bias, res_std = res_df[self.pSA_keys].mean(axis=0), res_df[
            self.pSA_keys
        ].std(axis=0)

        cv_bias_std = (
            res_df.groupby("cv_iter", observed=True)[self.pSA_keys].mean().std(axis=0)
        )
        self.ax1.fill_between(
            self.pSA_periods,
            model_bias[self.pSA_keys].values - cv_bias_std[self.pSA_keys].values,
            model_bias[self.pSA_keys].values + cv_bias_std[self.pSA_keys].values,
            **plt_kwargs,
        )

        cv_res_std_std = (
            res_df.groupby("cv_iter", observed=True)[self.pSA_keys].std().std(axis=0)
        )
        self.ax3.fill_between(
            self.pSA_periods,
            res_std[self.pSA_keys].values - cv_res_std_std[self.pSA_keys].values,
            res_std[self.pSA_keys].values + cv_res_std_std[self.pSA_keys].values,
            **plt_kwargs,
        )

        return self

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
        group_keys: list[str],
        group_colors: list,
        group_labels: list[str] | None = None,
        pSA_keys_periods: tuple[list[str], list[float]] | None = None,
        im_set: str = "pSA",
        figsize: tuple[float, float] = (16, 6),
        bias_ylim: tuple[float, float] = (-0.8, 0.8),
        std_ylim: tuple[float, float] = (0, 0.8),
        x_axis_limits: tuple[float, float] = (0.01, 10.0),
        **fig_kwargs,
    ):
        """
        Initializes the GroupedBiasStdPlot class.

        Parameters
        ----------
        group_key : str
            The key to group the results by.
        group_bin_edges : list[float]
            The bin edges for grouping.
        group_keys : list[str]
            The keys for each group.
        group_colors : list
            The colors for each group.
        pSA_keys_periods : tuple[list[str], list[float]], optional
            Tuple containing lists of pSA keys and their corresponding periods.
            If None, im_set is used to determine the pSA keys and periods.
        im_set : str, optional
            The IM set to use. Default is "pSA".
        """
        super().__init__(
            pSA_keys_periods,
            im_set,
            figsize,
            bias_ylim,
            std_ylim,
            x_axis_limits=x_axis_limits,
            **fig_kwargs,
        )

        self.group_key = group_key
        self.group_bin_key = f"{group_key}_bin"
        self.group_bin_edges = group_bin_edges
        self.group_keys = group_keys
        self.group_colors = group_colors
        self.group_labels = group_labels

    def add_grouped_results(
        self, res_df: pd.DataFrame, add_legend_entries: bool, **plt_kwargs
    ):
        """Groups the results and adds to the plot"""
        res_df = res_df.copy()
        res_df[self.group_bin_key] = pd.cut(
            res_df[self.group_key],
            bins=self.group_bin_edges,
            labels=self.group_keys,
            include_lowest=True,
        )

        self._plot_groups(res_df, self.group_bin_key, add_legend_entries, **plt_kwargs)
        return self

    def add_categorial_results(
        self,
        res_df: pd.DataFrame,
        add_legend_entries: bool,
        **plt_kwargs,
    ):
        """Adds categorial results (already grouped) to the plot"""
        self._plot_groups(res_df, self.group_key, add_legend_entries, **plt_kwargs)
        return self

    def _plot_groups(
        self, res_df: pd.DataFrame, key: str, add_legend_entries: bool, **plt_kwargs
    ):
        # Bias
        for i, cur_bin_label in enumerate(self.group_keys):
            cur_record_ids = res_df.index[res_df[key] == cur_bin_label]
            cur_model_bias = res_df.loc[cur_record_ids, self.pSA_keys].mean(axis=0)
            self.ax1.plot(
                self.pSA_periods,
                cur_model_bias[self.pSA_keys].values,
                c=self.group_colors[i],
                **plt_kwargs,
                label=(
                    f"{cur_bin_label if self.group_labels is None else self.group_labels[i]} (N={len(cur_record_ids):,})"
                    if add_legend_entries
                    else None
                ),
            )

        # Std
        for i, cur_bin_label in enumerate(self.group_keys):
            cur_record_ids = res_df.index[res_df[key] == cur_bin_label]
            cur_res_std = res_df.loc[cur_record_ids, self.pSA_keys].std(axis=0)
            self.ax3.plot(
                self.pSA_periods,
                cur_res_std[self.pSA_keys].values,
                c=self.group_colors[i],
                label=(
                    f"{cur_bin_label if self.group_labels is None else self.group_labels[i]} (N={len(cur_record_ids):,})"
                    if add_legend_entries
                    else None
                ),
                **plt_kwargs,
            )

    def add_nn_gmm_grouped_cv_band(self, res_df: pd.DataFrame, **plt_kwargs):
        """Adds grouped NN-GMM CV band to the plot"""
        res_df = res_df.copy()
        res_df[self.group_bin_key] = pd.cut(
            res_df[self.group_key],
            bins=self.group_bin_edges,
            labels=self.group_keys,
            include_lowest=True,
        )

        self._plot_nn_gmm_grouped_cv_band(res_df, self.group_bin_key, **plt_kwargs)
        return self

    def add_nn_gmm_categorial_cv_band(self, res_df: pd.DataFrame, **plt_kwargs):
        """Adds categorial NN-GMM CV band to the plot"""
        self._plot_nn_gmm_grouped_cv_band(res_df, self.group_key, **plt_kwargs)
        return self

    def _plot_nn_gmm_grouped_cv_band(
        self, res_df: pd.DataFrame, key: str, **plt_kwargs
    ):
        # Bias
        for i, cur_bin_label in enumerate(self.group_keys):
            cur_record_ids = res_df.index[res_df[key] == cur_bin_label]
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
        for i, cur_bin_label in enumerate(self.group_keys):
            cur_record_ids = res_df.index[res_df[key] == cur_bin_label]
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


def site_bias_histogram_comparison(
    model_dir_1: Path,
    model_dir_2: Path,
    output_dir: Path,
    ims: list[str] = constants.PLOT_IMS,
    n_bins: int = 50,
    dpi: int = 100,
):
    """
    Creates histogram comparison plots of
    site bias between two NN-GMM models.
    """
    ims = ims or constants.PLOT_IMS

    res_df_1, *_ = analysis.get_nn_sim_residuals(model_dir_1)
    res_df_2, *_ = analysis.get_nn_sim_residuals(model_dir_2)

    with data.DuckIMDB(
        nn_gmm.load_config(model_dir_1 / "run_config.yaml").imdb_ffp, readonly=True
    ) as imdb:
        site_df = imdb.get_site_df()

    site_bias_1, _ = analysis.get_site_bias_std(res_df_1, site_df)
    site_bias_2, _ = analysis.get_site_bias_std(res_df_2, site_df)

    bins = np.linspace(-0.75, 0.75, n_bins + 1)

    for im in ims:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        ax.hist(
            site_bias_1[im],
            bins=bins,
            color="b",
            alpha=0.5,
            label=f"{model_dir_1.name}, Mean={site_bias_1[im].mean():.3f}, Std={site_bias_1[im].std():.3f}",
            edgecolor="k",
        )
        ax.hist(
            site_bias_2[im],
            bins=bins,
            color="r",
            alpha=0.5,
            label=f"{model_dir_2.name}, Mean={site_bias_2[im].mean():.3f}, Std={site_bias_2[im].std():.3f}",
            edgecolor="k",
        )
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

        ax.set_xlabel("Site Bias")
        ax.set_ylabel("Count")
        ax.set_title(f"{utils.get_nice_im_name(im)}")
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / f"site_bias_comparison_{im}.png")
        plt.close(fig)

        mlt.utils.write_to_yaml(
            dict(
                type="site-bias-comparison",
                model_dir_1=model_dir_1.name,
                model_dir_2=model_dir_2.name,
                im=im,
                is_mera=False,
            ),
            output_dir / f"site_bias_comparison_{im}.yaml",
        )


def site_bias_res_std_comparison(
    model_dirs: list[Path], output_dir: Path, dpi: int = 100
):
    """Generates a site bias and site residual
    standard deviation comparison plot wrt. pSA"""
    fig, ax1, ax2, ax3, ax4 = plot_utils.get_bias_residual_fig(
        figsize=(16, 6),
        bias_y_axis_limits=(-0.05, 0.05),
        std_y_axis_limits=(0, 0.20),
        bias_y_label="Mean Site Bias",
        std_y_label="Site Bias Standard Deviation",
        dpi=dpi,
    )

    site_df = None
    for model_dir in model_dirs:
        if site_df is None:
            with data.DuckIMDB(
                nn_gmm.load_config(model_dir / "run_config.yaml").imdb_ffp,
                readonly=True,
            ) as imdb:
                site_df = imdb.get_site_df()

        res_df, *_ = analysis.get_nn_sim_residuals(model_dir)
        site_bias, _ = analysis.get_site_bias_std(res_df, site_df)

        ax1.plot(
            constants.PSA_PERIODS,
            np.mean(site_bias[constants.PSA_KEYS].values, axis=0),
            label=model_dir.name,
        )
        ax3.plot(
            constants.PSA_PERIODS,
            np.std(site_bias[constants.PSA_KEYS].values, axis=0),
            label=model_dir.name,
        )

    ax1.legend()
    fig.savefig(output_dir / "site_bias_res_std_comparison.png")
    plt.close(fig)

    mlt.utils.write_to_yaml(
        dict(
            type="site-bias-res-std-comparison",
            model_dirs=[d.name for d in model_dirs],
            is_mera=False,
        ),
        output_dir / "site_bias_res_std_comparison.yaml",
    )


def mera_basin_site_term_comparison(
    model_dir_1: Path,
    model_dir_2: Path,
    output_dir: Path,
):
    from mera import MeraResults

    # Check that MERA results exist
    if not (mera_dir_1 := model_dir_1 / "mera_site_term").exists():
        raise FileNotFoundError(f"MERA results directory not found: {mera_dir_1}")

    if not (mera_dir_2 := model_dir_2 / "mera_site_term").exists():
        raise FileNotFoundError(f"MERA results directory not found: {mera_dir_2}")

    run_config_1 = nn_gmm.load_config(model_dir_1 / "run_config.yaml")
    with data.DuckIMDB(
        run_config_1.imdb_ffp,
        readonly=True,
    ) as imdb:
        site_df = imdb.get_site_df().set_index("site_id")

    mera_result_1 = MeraResults.load_from_parquet(mera_dir_1)
    mera_result_2 = MeraResults.load_from_parquet(mera_dir_2)
    assert mera_result_1.bias_std_df.index.equals(
        mera_result_2.bias_std_df.index
    ), "IMs do not match between the two MERA results"
    ims = mera_result_1.bias_std_df.index.values
    periods = [utils.get_pSA_period(im) for im in ims]

    site_term_1 = mera_result_1.site_res_df
    site_term_1.loc[:, ["lon", "lat"]] = site_df.loc[site_term_1.index, ["lon", "lat"]]
    site_term_1 = utils.add_basin_column(site_term_1)
    site_term_2 = mera_result_2.site_res_df
    site_term_2.loc[:, ["lon", "lat"]] = site_df.loc[site_term_2.index, ["lon", "lat"]]
    site_term_2 = utils.add_basin_column(site_term_2)

    basin_labels = list(site_term_1["basin"].unique())
    basin_colors = sns.color_palette("tab10", len(basin_labels))

    bias_std_plot = (
        GroupedBiasStdPlot(
            "basin",
            None,
            basin_labels,
            basin_colors,
            pSA_keys_periods=(ims, periods),
            bias_ylim=(-0.2, 0.2),
            std_ylim=(0, 0.4),
        )
        # .add_results(site_term_1, linestyle="-", label="Model 1 - All Sites", c="k", linewidth=2.0)
        # .add_results(site_term_2, linestyle="--", label="Model 2 - All Sites", c="k", linewidth=2.0)
    )
    bias_std_plot.ax1.set_ylabel("Mean Basin Site Term")
    bias_std_plot.ax3.set_ylabel("Basin Site Term Standard Deviation")

    bias_std_plot.add_categorial_results(site_term_1, True)
    bias_std_plot.add_categorial_results(
        site_term_2,
        False,
        linestyle="--",
    )
    bias_std_plot.add_legend(bias_std_plot.ax3)

    bias_std_plot.fig.savefig(output_dir / "mera_basin_site_term_comparison.png")
    plt.close(bias_std_plot.fig)

    mlt.utils.write_to_yaml(
        dict(
            type="mera-basin-site-term-comparison",
            model_dir_1=model_dir_1.name,
            model_dir_2=model_dir_2.name,
            is_mera=True,
        ),
        output_dir / "mera_basin_site_term_comparison.yaml",
        clobber=True,
    )


def site_hazard(
    ds_results_dir: Path,
    output_dir: Path,
    emp_ds_results_dir: Path | None = None,
    lower_ds_results_dir: Path | None = None,
    upper_ds_results_dir: Path | None = None,
):
    """
    Lower/upper DS results directories are the lower/upper branch
    (e.g. 5th/95th quantile) NN-GMM DS hazard results, shown as a band.
    """
    if (lower_ds_results_dir is None) != (upper_ds_results_dir is None):
        raise ValueError(
            "lower_ds_results_dir and upper_ds_results_dir have to be specified together."
        )

    def load_ds_hazard(results_dir: Path | None) -> dict:
        if results_dir is None:
            return {}
        return {
            cur_ffp.stem: pd.read_pickle(cur_ffp)
            for cur_ffp in results_dir.glob("*.pkl")
        }

    # Load Cybershake fault hazard
    cs_flt_hazard = pd.read_pickle(
        constants.HAZARD_RESOURCES_DIR / "flt/Cybershake_hazard_data.pkl"
    )

    nn_ds_hazard = load_ds_hazard(ds_results_dir)
    emp_ds_hazard = load_ds_hazard(emp_ds_results_dir)
    lower_ds_hazard = load_ds_hazard(lower_ds_results_dir)
    upper_ds_hazard = load_ds_hazard(upper_ds_results_dir)

    for site, hazard_result in nn_ds_hazard.items():
        logger.info(f"Creating hazard plots for site: {site}")
        for cur_im in constants.PLOT_IMS:
            _create_hazard_plot(
                hazard_result,
                site,
                cur_im,
                output_dir / f"{site}_hazard_{cur_im}.png",
                cs_flt_hazard=cs_flt_hazard[cur_im],
                emp_ds_hazard=emp_ds_hazard.get(site),
                lower_nn_ds_hazard=lower_ds_hazard.get(site),
                upper_nn_ds_hazard=upper_ds_hazard.get(site),
            )


def _create_hazard_plot(
    nn_ds_hazard: dict[str, dict[str, pd.Series]],
    site: str,
    im: str,
    output_ffp: Path,
    cs_flt_hazard: pd.DataFrame = None,
    emp_ds_hazard: dict[str, pd.Series] | None = None,
    lower_nn_ds_hazard: dict[str, dict[str, pd.Series]] | None = None,
    upper_nn_ds_hazard: dict[str, dict[str, pd.Series]] | None = None,
    dpi: int = 300,
    figsize: tuple = (8, 6),
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im_levels = nn_ds_hazard["total"][im].index.values

    ax.plot(im_levels, nn_ds_hazard["total"][im].values, label="NN-GMM DS", color="b")
    if lower_nn_ds_hazard is not None and upper_nn_ds_hazard is not None:
        assert np.allclose(
            lower_nn_ds_hazard["total"][im].index.values, im_levels
        ) and np.allclose(
            upper_nn_ds_hazard["total"][im].index.values, im_levels
        ), "Lower/upper DS hazard IM levels do not match NN-GMM DS hazard IM levels"
        ax.fill_between(
            im_levels,
            lower_nn_ds_hazard["total"][im].values,
            upper_nn_ds_hazard["total"][im].values,
            color="b",
            alpha=0.2,
            linewidth=0,
            label="NN-GMM DS Epistemic Band",
        )
    # ax.plot(im_levels, nn_ds_hazard["crustal"][im].values, label="NN-GMM DS Crustal", color="b", linestyle="--")
    # ax.plot(im_levels, nn_ds_hazard["subduction_slab"][im].values, label="NN-GMM DS Subduction", color="b", linestyle=":")

    if cs_flt_hazard is not None:
        ax.plot(
            cs_flt_hazard.columns.values,
            cs_flt_hazard.loc[site].values,
            label="Cybershake Fault Hazard",
            color="k",
        )

    if emp_ds_hazard is not None:
        assert np.allclose(
            emp_ds_hazard["total"][im].index.values, im_levels
        ), "Empirical DS hazard IM levels do not match NN-GMM DS hazard IM levels"
        ax.plot(
            im_levels,
            emp_ds_hazard["total"][im].values,
            label="Empirical DS",
            color="g",
        )
        # ax.plot(
        #     im_levels,
        #     emp_ds_hazard["crustal"][im].values,
        #     label="Empirical DS Crustal",
        #     color="g",
        #     linestyle="--",
        # )
        # ax.plot(
        #     im_levels,
        #     emp_ds_hazard["subduction"][im].values,
        #     label="Empirical DS Subduction",
        #     color="g",
        #     linestyle=":",
        # )

    ax.set_xlabel(f"{im} (g)")
    ax.set_ylabel("Annual Exceedance Probability")
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 10)
    ax.set_xlim(im_levels.min(), im_levels.max())
    ax.legend()

    ax.text(
        0.5,
        0.975,
        site,
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        dict(
            type="site-hazard-plot",
            site=site,
            im=im,
        ),
        output_ffp.with_suffix(".yaml"),
        clobber=True,
    )


def site_uhs(
    ds_results_dir: Path,
    output_dir: Path,
    rps: list[float | int],
    emp_ds_results_dir: Path | None = None,
):
    sites = [item.stem for item in ds_results_dir.glob("*.pkl")]

    # Cybershake fault UHS
    cs_flt_uhs = hazard.compute_cs_flt_uhs(sites, rps)

    # NN-GMM DS UHS
    nn_ds_uhs = hazard.compute_nn_ds_uhs(ds_results_dir, rps, sites=sites)

    # Empirical DS UHS
    emp_ds_uhs = None
    if emp_ds_results_dir is not None:
        emp_ds_uhs = hazard.compute_emp_ds_uhs(emp_ds_results_dir, rps, sites=sites)

    # Create UHS plots
    logger.info(f"Creating UHS plots for {len(sites)} sites")
    assert cs_flt_uhs.coords["im"].values.tolist() == constants.PSA_KEYS
    assert nn_ds_uhs.coords["im"].values.tolist() == constants.PSA_KEYS
    for site in sites:
        for rp in rps:
            output_ffp = output_dir / f"{site}_uhs_rp{int(rp)}.png"

            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

            ax.plot(
                constants.PSA_PERIODS,
                cs_flt_uhs.sel(site=site, rp=rp),
                label="Cybershake Fault Hazard",
                color="k",
            )
            ax.plot(
                constants.PSA_PERIODS,
                nn_ds_uhs.sel(site=site, rp=rp),
                label="NN-GMM DS Hazard",
                color="b",
            )

            if emp_ds_uhs is not None:
                ax.plot(
                    constants.PSA_PERIODS,
                    emp_ds_uhs.sel(site=site, rp=rp),
                    label="Empirical DS Hazard",
                    color="g",
                )

            ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
            ax.set_xlabel("Period (s)")
            ax.set_ylabel("pSA (g)")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlim(0.01, 10)
            ax.set_ylim(1e-4, 10)
            ax.legend(loc="lower left")

            ax.text(
                0.5,
                0.975,
                f"{site}, {int(rp)}-year RP",
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="top",
                fontweight="bold",
            )

            fig.tight_layout()
            fig.savefig(output_dir / output_ffp)
            plt.close(fig)

            mlt.utils.write_to_yaml(
                dict(
                    type="site-uhs-plot",
                    site=site,
                    rp=rp,
                ),
                output_ffp.with_suffix(".yaml"),
                clobber=True,
            )


def pred_vs_res_std(model_dir: Path):
    """
    Creates a predicted vs residual standard deviation plot
    with respect to pSA
    """
    run_config = nn_gmm.load_config(model_dir / "run_config.yaml")

    res_df, pred_df, *_ = analysis.get_nn_sim_residuals(model_dir)

    res_stds = res_df.groupby("cv_iter")[run_config.ims].std()
    res_stds_mean = res_stds.mean()
    res_stds_std = res_stds.std()

    pred_stds = pred_df.groupby("cv_iter")[run_config.pred_std_keys].mean()
    pred_stds.columns = run_config.ims
    pred_stds_mean = pred_stds.mean()
    pred_stds_std = pred_stds.std()

    assert np.all(res_stds.columns == pred_stds.columns)

    output_ffp = model_dir / "plots/pred_vs_res_std.png"
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        run_config.pSA_periods, pred_stds_mean.values, label="Predicted Std", color="b"
    )
    ax.fill_between(
        run_config.pSA_periods,
        pred_stds_mean.values - pred_stds_std.values,
        pred_stds_mean.values + pred_stds_std.values,
        color="b",
        alpha=0.5,
    )

    ax.plot(
        run_config.pSA_periods, res_stds_mean.values, label="Residual Std", color="r"
    )
    ax.fill_between(
        run_config.pSA_periods,
        res_stds_mean.values - res_stds_std.values,
        res_stds_mean.values + res_stds_std.values,
        color="r",
        alpha=0.5,
    )

    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Standard Deviation")
    ax.set_xlim(0.01, 10)
    ax.set_ylim(0, 0.6)
    ax.set_xscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        dict(
            type="pred-vs-res-std-plot",
        ),
        model_dir / "plots/pred_vs_res_std.yaml",
        clobber=True,
    )


def feature_importance_plots(
    shap_values: shap.Explanation, run_config: nn_gmm.BaseRunConfig, output_dir: Path
):
    """
    Creates feature importance plots based on SHAP values
    """
    # Bar feature importance
    n_ims = len(run_config.pred_mean_keys)
    for i in range(n_ims):
        mean_shap_values = shap_values[:, :, i]
        std_shap_values = shap_values[:, :, i + n_ims]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))

        shap.plots.bar(mean_shap_values, ax=ax1, show=False, max_display=20)
        shap.plots.bar(std_shap_values, ax=ax2, show=False, max_display=20)

        out_ffp = output_dir / f"shap_feature_importance_{run_config.ims[i]}.png"
        fig.savefig(out_ffp)
        plt.close(fig)

        mlt.utils.write_to_yaml(
            dict(
                type="shap-feature-importance",
                im=str(run_config.ims[i]),
            ),
            out_ffp.with_suffix(".yaml"),
            clobber=True,
        )


def bias_res_std_tect_type(cv_results_dir: Path, output_dir: Path):
    """
    Creates a bias and residual standard deviation
    plot grouped by tectonic type
    """
    run_config = nn_gmm.load_config(cv_results_dir / "run_config.yaml")

    res_df, pred_df, sim_df, record_info_df = analysis.get_nn_sim_residuals(
        cv_results_dir
    )

    with data.DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        event_df = imdb.get_event_df()
    tect_types = event_df["tect_type"].unique().tolist()
    tect_colors = sns.color_palette("Set1", len(tect_types))

    res_df["tect_type"] = event_df.loc[res_df.event_int_id, "tect_type"].values

    bias_std_plot = GroupedBiasStdPlot(
        "tect_type",
        None,
        tect_types,
        tect_colors,
        pSA_keys_periods=(run_config.ims, run_config.pSA_periods),
        bias_ylim=(-0.2, 0.2),
        std_ylim=(0, 0.6),
    )
    bias_std_plot.add_categorial_results(res_df, add_legend_entries=True)
    bias_std_plot.add_legend()

    output_ffp = output_dir / "bias_resStd_tectType.png"
    bias_std_plot.fig.savefig(output_ffp)
    plt.close(bias_std_plot.fig)


def nn_fault_hazard_bias_res_std(
    nn_flt_hazard_results_ffp: Path,
    output_dir: Path,
    rps: list[int],
    loc_nn_flt_hazard_results_ffp: Path | None = None,
    cs_parametric_flt_hazard_ffp: Path | None = None,
):
    cs_hazard = pd.read_pickle(constants.CS_FLT_HAZARD_FFP)
    base_nn_hazard = pd.read_pickle(nn_flt_hazard_results_ffp)
    _, base_nn_mean_residual, base_nn_residual_std = hazard.compute_fault_nn_hazard_error(
        base_nn_hazard, cs_hazard, rps
    )

    cs_param_hazard = None
    if cs_parametric_flt_hazard_ffp is not None:
        cs_param_hazard = pd.read_pickle(cs_parametric_flt_hazard_ffp)
        _, cs_param_mean_residual, cs_param_residual_std = (
            hazard.compute_fault_nn_hazard_error(cs_param_hazard, cs_hazard, rps)
        )

    loc_nn_hazard = None
    if loc_nn_flt_hazard_results_ffp is not None:
        loc_nn_hazard = pd.read_pickle(loc_nn_flt_hazard_results_ffp)
        _, loc_nn_mean_residual, loc_nn_residual_std = (
            hazard.compute_fault_nn_hazard_error(loc_nn_hazard, cs_hazard, rps)
        )

    bias_std_plot = BiasStdPlot(
        figsize=constants.FIG_SIZE,
        dpi=constants.FIG_DPI,
        bias_ylim=(-0.5, 0.5),
        std_ylim=(0, 0.95),
        main_wspace=0.175,
        left=0.07,
    )

    linestyles = ["-", "--", "dashdot"]
    for i, rp in enumerate(rps):
        bias_std_plot.ax1.plot(
            constants.PSA_PERIODS,
            base_nn_mean_residual.loc[constants.PSA_KEYS, rp].values,
            c="blue",
            linestyle=linestyles[i],
            linewidth=constants.FIG_LINEWIDTH,
            label=f"RP={rp} (Base Surrogate)",
        )
        if cs_param_hazard is not None:
            bias_std_plot.ax1.plot(
                constants.PSA_PERIODS,
                cs_param_mean_residual.loc[constants.PSA_KEYS, rp].values,
                c="black",
                linestyle=linestyles[i],
                linewidth=constants.FIG_LINEWIDTH,
                label=f"RP={rp} (CS Parametric)",
            )
        if loc_nn_hazard is not None:
            bias_std_plot.ax1.plot(
                constants.PSA_PERIODS,
                loc_nn_mean_residual.loc[constants.PSA_KEYS, rp].values,
                c="red",
                linestyle=linestyles[i],
                linewidth=constants.FIG_LINEWIDTH,
                label=f"RP={rp} (Loc Surrogate)",
            )

        bias_std_plot.ax3.plot(
            constants.PSA_PERIODS,
            base_nn_residual_std.loc[constants.PSA_KEYS, rp].values,
            c="blue",
            linestyle=linestyles[i],
            linewidth=constants.FIG_LINEWIDTH,
            label=f"RP={rp}",
        )
        if cs_param_hazard is not None:
            bias_std_plot.ax3.plot(
                constants.PSA_PERIODS,
                cs_param_residual_std.loc[constants.PSA_KEYS, rp].values,
                c="black",
                linestyle=linestyles[i],
                linewidth=constants.FIG_LINEWIDTH,
                label=f"RP={rp} (CS Parametric)",
            )
        if loc_nn_hazard is not None:
            bias_std_plot.ax3.plot(
                constants.PSA_PERIODS,
                loc_nn_residual_std.loc[constants.PSA_KEYS, rp].values,
                c="red",
                linestyle=linestyles[i],
                linewidth=constants.FIG_LINEWIDTH,
                label=f"RP={rp} (Loc NN)",
            )

    bias_std_plot.ax1.set_ylabel("Mean hazard residual")
    bias_std_plot.ax1.legend()

    bias_std_plot.fig.savefig(
        output_dir / "nn_fault_hazard_bias_res_std.png",
        dpi=constants.FIG_DPI,
    )
    plt.close(bias_std_plot.fig)


def nn_mera_bias_res_std(model_dir: Path):
    """
    Creates bias and residual standard deviation
    plot based on MERA results for the NN-GMM model
    """
    from mera import MeraResults

    assert (
        mera_dir := model_dir / "mera_site_term"
    ).exists(), "MERA results not found in model directory"

    mera_results = MeraResults.load_from_parquet(mera_dir)
    periods = [
        utils.get_pSA_period(im)
        for im in mera_results.bias_std_df.index.values.astype(str)
    ]

    bias_std_plot = BiasStdPlot(
        figsize=constants.FIG_SIZE,
        dpi=constants.FIG_DPI,
        bias_ylim=(-0.5, 0.5),
        std_ylim=(0, 0.95),
        main_wspace=0.175,
        left=0.07,
    )

    bias_std_plot.ax1.set_ylabel("Model prediction bias, a", labelpad=-2)
    bias_std_plot.ax3.set_ylabel(r"Residual Standard Deviation, $\sigma$")

    bias_std_df = mera_results.bias_std_df.copy()
    bias_std_df.index = periods

    # Bias
    bias_std_plot.add_bias(
        bias_std_df["bias"],
        c="blue",
        linestyle="-",
        linewidth=constants.FIG_LINEWIDTH,
    )

    bias_std_plot.add_std(
        bias_std_df["sigma"],
        c="blue",
        label=r"Total, $\sigma$",
        linestyle="solid",
        linewidth=constants.FIG_LINEWIDTH,
    )
    bias_std_plot.add_std(
        bias_std_df["tau"],
        c="purple",
        label=r"Between-event, $\tau$",
        linestyle="solid",
        linewidth=constants.FIG_GROUP_LINEWIDTH,
    )
    bias_std_plot.add_std(
        bias_std_df["phi_S2S"],
        c="red",
        linestyle="solid",
        label=r"Site-to-site, $\phi_{S2S}$",
        linewidth=constants.FIG_GROUP_LINEWIDTH,
    )
    bias_std_plot.add_std(
        bias_std_df["phi_w"],
        c="green",
        linestyle="solid",
        label=r"Remaining, $\phi_w$",
        linewidth=constants.FIG_GROUP_LINEWIDTH,
    )

    bias_std_plot.add_legend(bias_std_plot.ax3)
    bias_std_plot.ax3.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    output_ffp = model_dir / "plots/nn_mera_bias_res_std.png"
    bias_std_plot.fig.savefig(
        output_ffp,
        dpi=constants.FIG_DPI,
    )

    # Metadata
    mlt.utils.write_to_yaml(
        dict(
            type="mera_model_bias_std",
            model=str(model_dir.name),
            is_mera=True,
        ),
        output_ffp.with_suffix(".yaml"),
        clobber=True,
    )


def fault_hazard(
    cs_flt_hazard: dict,
    nn_flt_hazard: dict,
    site: str,
    im: str,
    output_ffp: Path,
    cs_parametric_flt_hazard: dict | None = None,
):
    """Plot fault hazard curves for a given site and IM."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        cs_flt_hazard[im].loc[site].index.values,
        cs_flt_hazard[im].loc[site].values,
        label="Cybershake",
        color="black",
    )
    ax.plot(
        nn_flt_hazard[im].loc[site].index.values,
        nn_flt_hazard[im].loc[site].values,
        label="NN-GMM",
        color="blue",
    )
    if cs_parametric_flt_hazard is not None:
        ax.plot(
            cs_parametric_flt_hazard[im].loc[site].index.values,
            cs_parametric_flt_hazard[im].loc[site].values,
            label="Cybershake Parametric",
            color="red",
        )
    ax.set_xlabel(f"{im} (g)")
    ax.set_ylabel("Annual Exceedance Probability")
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 10)
    ax.set_xlim(
        nn_flt_hazard[im].loc[site].index.min(),
        nn_flt_hazard[im].loc[site].index.max(),
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_ffp, dpi=300)
    plt.close(fig)
