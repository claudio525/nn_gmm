import gc
import tempfile
from typing import Tuple, Iterable, Callable, Dict, List, Any, Union
from pathlib import Path
from collections import namedtuple

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib

import empirical.util.classdef as classdef
import empirical.util.empirical_factory as emp_factory
from visualization.gmt.plotting import plot_multiple, plot_single

from nn_gmm.src.model import GMM
from nn_gmm.src.utils import get_station_from_id, get_station_lookup, to_path, to_list
from nn_gmm.src import data
from nn_gmm.src.eval import get_realisation_residuals
from .plotting_funcs import *


class EventPlotGen(ModelEventBasePlotGen):
    """Class for generating event specific plots"""

    def __init__(
        self, plot_items_ffp: str, model: GMM, data_dirs: List[Path], output_dir: Path
    ):
        super().__init__(plot_items_ffp, model, data_dirs, output_dir)

    def gen_IM_feature_plots(
        self, events: Union[str, List[str]], ims: List[str], feature_key: str
    ):
        """Generates IM vs feature plots for the specified IMs and events"""
        events = [events] if isinstance(events, str) else events

        for event in events:
            for im in ims:
                self.gen_IM_feature_plot(event, im, feature_key)

    def gen_IM_residuals_plot(
        self, event: str, im: str, feature_key: str, log_space: bool = False
    ):
        """Creates a residual sactter plot for the specified IM
        against the feature of interest"""
        event_out_dir = self.output_dir / event
        if not event_out_dir.is_dir():
            event_out_dir.mkdir(parents=True)

        sim_df, mean_df, std_df = self._get_event_estimates(event)
        assert np.all(sim_df.index == mean_df.index)

        im_name = im.replace(".", "p")
        if log_space:
            residual = sim_df.loc[:, im] - mean_df.loc[:, im].apply(np.log)
            title = f"{im_name} Log ratio - Mean absolute log ratio {np.mean(np.abs(residual.values)):.4f}"
        else:
            residual = sim_df.loc[:, im].apply(np.exp) - mean_df.loc[:, im]
            title = f"{im_name} - Mean residual {np.mean(np.abs(residual.values)):.4f}"

        filename = (
            f"{im}_{feature_key}_residual.png"
            if not log_space
            else f"{im}_{feature_key}_log_ratio.png"
        )

        residual_hist_plot(
            sim_df.loc[:, feature_key].values,
            residual.values,
            feature_key,
            im,
            output_ffp=str(event_out_dir / filename),
            title=title,
        )

    def gen_IM_feature_plot(self, event: str, im: str, feature_key: str):
        """Generates a IM vs feature plot for the specified IM and event"""
        event_out_dir = self.output_dir / event
        if not event_out_dir.is_dir():
            event_out_dir.mkdir(parents=True)

        sim_df, mean_df, std_df = self._get_event_estimates(event)
        assert np.all(sim_df.index == mean_df.index)

        fig = plt.figure(figsize=(18, 13.5))
        # fig = plt.figure(figsize=(7, 5.5))

        x, sim_y = sim_df[feature_key], sim_df[im].apply(np.exp)
        plt.scatter(x, sim_y, label="Simulation", s=1.0)
        plt.scatter(x, mean_df[im], label="Estimated", s=1.0)

        # Aggregate at each unique value of the feature of interest
        mean_df[feature_key] = sim_df[feature_key]
        if std_df is not None:
            std_df[feature_key] = sim_df[feature_key]

        bin_edges = np.linspace(x.min() - 1e-5, x.max(), 50)
        bin_indices = np.digitize(sim_df[feature_key].values, bin_edges, right=True)

        # Get center points of the bins (there might be a better way of doing this
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        mean_df["bin_ix"] = bin_indices
        bin_mean_mean = mean_df.groupby("bin_ix").mean()[im]

        if std_df is not None:
            std_df["bin_ix"] = bin_indices
            bin_mean_std = std_df.groupby("bin_ix").mean()[im]

        plt.plot(
            bin_centers[bin_mean_mean.index - 1],
            bin_mean_mean,
            c="k",
            marker="o",
            ms=1.5,
            linewidth=1.25,
            label="Bin mean of estimated means",
        )
        if std_df is not None:
            plt.plot(
                bin_centers[bin_mean_std.index - 1],
                bin_mean_mean * np.exp(bin_mean_std),
                linestyle="--",
                c="k",
                marker="o",
                ms=1.75,
                linewidth=1.25,
                label="Bin mean of estiamted stds",
            )
            plt.plot(
                bin_centers[bin_mean_std.index - 1],
                bin_mean_mean * np.exp(-bin_mean_std),
                linestyle="--",
                c="k",
                marker="o",
                ms=1.75,
                linewidth=1.25,
            )

        plt.xlabel(feature_key)
        plt.ylabel(im)
        plt.yscale("log")
        plt.grid(which="major", linestyle="-", linewidth=0.5)
        plt.grid(which="minor", linestyle="--", linewidth=0.25)
        plt.title("{} {} {}".format(event, im, feature_key))
        plt.legend()
        set_plot_lims(x, [sim_y, mean_df[im]])

        output_ffp = self.output_dir / event / f"{event}_{im}_{feature_key}.png"
        plt.savefig(output_ffp)
        plt.close()

    def plot_spatial_events_maps(
        self,
        events: List[str],
        ims: List[str],
        data_type: str = "est_mean",
        events_cb_options: Dict[str, Dict[str, Dict]] = None,
        suffix: str = "",
        n_procs: int = 4,
    ):
        """Plots spatial maps for the specified IMs and events
        See gen_event_map_data for parameter details
        """
        result_cb_options = {}
        for event in events:
            print(f"Plotting spatial event maps for {event}")
            cur_cb_options = self.plot_spatial_event_maps(
                event,
                ims,
                data_type=data_type,
                cb_options=events_cb_options.get(event)
                if events_cb_options is not None
                else None,
                suffix=suffix,
                n_procs=n_procs,
            )
            result_cb_options[event] = cur_cb_options

        return result_cb_options

    def plot_spatial_event_maps(
        self,
        event: str,
        ims: List[str],
        data_type: str = "est_mean",
        cb_options: Dict[str, Dict] = None,
        suffix: str = "",
        n_procs: int = 4,
    ):
        """Plots spatial maps for the specified IMs and event
        See gen_event_map_data for parameter details
        """
        csv_ffps, result_cb_option = [], {}
        for im in ims:
            cur_csv, cur_cb_options = self.gen_event_map_data(
                event, im, data_type=data_type, cb_options=cb_options, suffix=suffix
            )
            csv_ffps.append(cur_csv)
            result_cb_option[im] = cur_cb_options

        # Generate the plots
        gmt_options = (
            DEFAULT_RES_GEN_GMT_PLOT_OPTIONS
            if "res" in data_type
            else DEFAULT_STANDARD_GMT_PLOT_OPTIONS
        )
        plot_multiple(
            self.plot_items_ffp, gmt_options, in_ffps=csv_ffps, n_procs=n_procs
        )

        return result_cb_option

    def plot_spatial_event_map(
        self, event: str, im: str, data_type: str = "est_mean", suffix: str = ""
    ):
        """plots spatial maps for the specified IM and event
        See gen_event_map_data for parameter details
        """
        # Create the data
        plot_csv_ffp, cb_options = self.gen_event_map_data(
            event, im, data_type=data_type, suffix=suffix
        )

        # Generate the plot
        plot_multiple(
            self.plot_items_ffp,
            DEFAULT_STANDARD_GMT_PLOT_OPTIONS,
            in_ffps=[str(plot_csv_ffp)],
        )

        return cb_options

    def _get_event_data(
        self, event: str, im: str, data_type: str = "est_mean", cb_options: Dict = None,
    ):
        """Gets the data for the speicified event, IMs and data type
        See gen_event_map_data for parameter details
        """
        sim_df, mean_est, std_est = self._get_event_estimates(event)
        assert np.all(mean_est.index.values == sim_df.index.values)

        n_std = 2.5
        sim_df["station"] = get_station_from_id(sim_df.index.values.astype(str))
        if "res" not in data_type:
            if data_type == "sim" or data_type == "est_mean":
                sim_im_df = sim_df[im].apply(np.exp).to_frame()
                data_df = (
                    mean_est.copy() if data_type == "est_mean" else sim_im_df.copy()
                )

                sim_im_df["station"] = sim_df.station
                cb_df = sim_im_df.groupby("station").mean()
            elif data_type.lower() == "est_std":
                data_df = std_est.copy()
                cb_df = sim_df.loc[:, [im, "station"]].groupby("station").std()
                n_std = 3.0
            else:
                raise ValueError(f"Invalid data_type: {data_type}")

            cb_options = (
                compute_GMT_std_ticks(cb_df[im], n_std=n_std, non_negative=True)
                if cb_options is None
                else cb_options[im]
            )
        elif data_type.lower() == "res_mean":
            data_df = (mean_est[im] / sim_df[im].apply(np.exp)).apply(np.log).to_frame()
            cb_options = {}
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        data_df = pd.merge(
            data_df,
            sim_df.loc[:, ["lat", "lon", "station"]],
            left_index=True,
            right_index=True,
            how="inner",
        )

        assert data_df.shape[0] == mean_est.shape[0]

        return data_df, cb_options

    def gen_event_map_data(
        self,
        event: str,
        im: str,
        data_type: str = "est_mean",
        cb_options: Dict = None,
        suffix: str = "",
    ):
        """Generates event based map data

        Realisations are aggregated at each station using the mean

        Parameters
        ----------
        event: str
            Event of interest
        im: str
            IM of interest
        data_type: str, optional
            The type of data to generate, has to be one of:
            "est_mean": Estimated mean from the NN GMM
            "est_std": Estimated std from the NN GMM
            "sim": Simulation IM values
            "res_mean": Residual between simulation IMs and
                estimated mean value from the NN GMM
        cb_options: Dict, optional
            GMT colour-bar options, if not given then
            these will be calculated based on the simulation data
            as this is consistent across models
            Useful when wanting two plots with the
            same colour-bar scale

        suffix: str, optional
            Filename suffix

        Returns
        -------
        str
            Path to the data csv
        dict
            The colour-bar options used
        """
        data_type = data_type.lower()
        event_out_dir = self.output_dir / event
        if not event_out_dir.is_dir():
            event_out_dir.mkdir(parents=True)

        data_df, cb_options = self._get_event_data(event, im, data_type, cb_options)

        # Aggregate across realisation at each station
        agg_df = data_df.groupby("station").mean()
        assert np.all(~agg_df[im].isna())

        im_name = im.replace(".", "p")
        gmt_options = get_gmt_options_dict(
            options={
                **{
                    "title": f"{im_name}-{event}-{data_type}",
                    "xyz-cpt-labels": f"{im}",
                },
                **cb_options,
            }
        )

        suffix = suffix if len(suffix) == 0 else f"_{suffix}"
        plot_csv_ffp = event_out_dir / f"{im_name}_{data_type}{suffix}"
        plot_csv_ffp = _gmt_save(agg_df, im, str(plot_csv_ffp), gmt_options=gmt_options)

        return plot_csv_ffp, cb_options

    def plot_realisation_maps(
        self,
        events: Union[str, List[str]],
        ims: Union[str, List[str]],
        data_types: Union[str, List[str]] = "est_mean",
        n_procs: int = 8,
        n_rels: int = None,
    ):
        # Generate the required data files
        for cur_data_type in to_list(data_types):
            plot_csv_ffps = []
            for cur_event in to_list(events):
                for cur_im in to_list(ims):
                    cur_plot_csv_ffps, cb_options = self.gen_rel_map_data(
                        cur_event, cur_im, data_type=cur_data_type, n_rels=n_rels
                    )
                    plot_csv_ffps.extend(cur_plot_csv_ffps)

            plot_multiple(
                self.plot_items_ffp,
                PLOT_TYPE_OPTIONS_MAPPING[cur_data_type],
                in_ffps=plot_csv_ffps,
                n_procs=n_procs,
            )

        return

    def gen_rel_map_data(
        self,
        event: str,
        im: str,
        data_type: str = "est_mean",
        cb_options: Dict = None,
        n_rels: int = None,
    ):
        """Generates map data for all realisation for the specified event (and IM)"""
        data_type = data_type.lower()
        event_out_dir = self.output_dir / event / "realisations"
        if not event_out_dir.is_dir():
            event_out_dir.mkdir(parents=True)

        data_df, cb_options = self._get_event_data(event, im, data_type, cb_options)

        data_df["rel"] = np.stack(np.char.split(data_df.index.values.astype(str), "_"))[
            :, 1
        ]

        # Only plot specified number of realisation
        rels = np.unique(data_df.rel)
        if n_rels is not None:
            rels = rels[: min(n_rels, len(rels))]

        plot_csv_ffps = []
        for cur_rel in rels:
            im_name = im.replace(".", "p")
            gmt_options = get_gmt_options_dict(
                options={
                    **{
                        "title": f"{im_name}-{event}-{cur_rel}-{data_type}",
                        "xyz-cpt-labels": f"{im}",
                    },
                    **cb_options,
                }
            )
            cur_plot_csv_ffp = _gmt_save(
                data_df.loc[data_df.rel == cur_rel],
                im,
                str(event_out_dir / f"{im_name}_{data_type}_{cur_rel}"),
                gmt_options=gmt_options,
            )
            plot_csv_ffps.append(cur_plot_csv_ffp)

        return plot_csv_ffps, cb_options