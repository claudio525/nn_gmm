from typing import Tuple, Iterable, Callable, Dict, List, Any, Union
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import empirical.util.classdef as classdef
import empirical.util.empirical_factory as emp_factory
from visualization.gmt.plotting import plot_multiple

from .model import GMM
from .utils import get_station_from_id, get_station_lookup
from . import data


IM_MEAN_KEY, IM_STD_KEY = "{}_mean", "{}_std"
TEMPLATE_OPTIONS_DICT = {"flags": [], "options": {}}


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

DEFAULT_RES_GEN_GMT_PLOT_OPTIONS = {
    "flags": ["xyz-grid", "xyz-landmask", "xyz-grid-contours"],
    "options": {
        "xyz-grid-search": "12m",
        "xyz-grid-automask": "12k",
        "xyz-cpt": "polar",
        "xyz-cpt-bg": "0/0/80",
        "xyz-cpt-fg": "80/0/0",
        "xyz-transparency": "30",
        "xyz-size": "1k",
        "xyz-cpt-inc": "0.125",
        "xyz-cpt-tick": "0.25",
        "xyz-cpt-min": "-1.0",
        "xyz-cpt-max": "1.0",
    },
}

DEFAULT_STANDARD_GMT_PLOT_OPTIONS = {
    "flags": ["xyz-grid", "xyz-landmask", "xyz-grid-contours", "xyz-cpt-invert"],
    "options": {
        "xyz-grid-search": "12m",
        "xyz-grid-automask": "12k",
        "xyz-cpt": "hot",
        "xyz-transparency": "30",
        "xyz-size": "1k",
        "xyz-cpt-min": "0",
        "xyz-cpt-max": "0.6",
    },
}


def multi_fig(
    ind_fig_size: Tuple[float, float], n_rows: int, n_cols: int, dpi: int = 100
) -> plt.Figure:
    fig_size = (ind_fig_size[0] * n_cols, ind_fig_size[1] * n_rows)
    return plt.figure(figsize=fig_size, dpi=dpi)


def create_multi_hist(
    df: pd.DataFrame,
    plot_ffp: str,
    title: str = None,
    bins=None,
    n_cols: int = 4,
    ind_fig_size: Tuple[int, int] = (25.0, 10.0),
):
    """Creates a single large figure with histograms plots for
    each of the columns in the provided dataframe
    """
    n_plots = len(df.columns)
    n_rows = np.ceil(n_plots / n_cols)

    fig = multi_fig(ind_fig_size, n_rows, n_cols)
    for ix, cur_col in enumerate(df.columns):
        ax = fig.add_subplot(n_rows, n_cols, ix + 1)
        sns.distplot(df.loc[:, cur_col], bins=bins, ax=ax, rug=False, kde=False)

    fig.suptitle(title)
    fig.savefig(plot_ffp)


class IMvsPlotGen:
    """Computes IM vs features (such as Mw, Rrup, vs30) plots"""

    CONST_DEFAULT_VALUES = pd.Series(
        data={
            "vs30": 388,
            "z1p0": 0.21,
            "z2p5": 1.3,
            "dip": 60,
            "rake": 45,
            "width": 18.2775,
            "ztor": 0,
            "mag": 7.0,
            "rjb": 91,
            "rrup": 91,
            "rx": 91,
            "ry": 91,
            "tect_type": "ACTIVE_SHALLOW",
        }
    )

    def __init__(self, model: GMM):
        self.model = model

    def get_B13_values(
        self,
        im: str,
        feature_key: str,
        feature_values: np.ndarray,
        const_values: pd.Series,
    ) -> Union[pd.DataFrame, None]:
        """Computes the IM values using the Bradley 2013 GMPE model,
        with one feature being varied and all others being kept constant

        Parameters
        ----------
        im: str
            The IM of interest
        feature_key: str, key
            The feature to vary
        feature_values: numpy array of floats
            The feature values at which to compute the IM value
        const_values: series
            The constant feature values to use

        Returns
        -------
        dataframe
            index: varied feature values
            columns: mu and sigma
        """
        if im.lower() not in ["pga", "pgv"] and not im.lower().startswith("psa"):
            return None

        period = 0 if im.lower() in ["pga", "pgv"] else float(im.split("_")[-1])
        im_values = []
        im_sigmas = []
        if feature_key == "rrup":
            fault = classdef.Fault(
                Mw=const_values.mag,
                rake=const_values.rake,
                dip=const_values.dip,
                ztor=const_values.ztor,
            )

            for cur_rrup in feature_values:
                cur_site = classdef.Site(
                    rrup=cur_rrup,
                    rjb=cur_rrup,
                    rx=cur_rrup,
                    hw=True,
                    rtvz=0,
                    vs30=const_values.vs30,
                    vs30measured=False,
                )
                emp_result = emp_factory.compute_gmm(
                    fault, cur_site, classdef.GMM.Br_10, im, [period]
                )
                if period == 0:
                    cur_mean, (cur_sigma, _, __) = emp_result
                else:
                    cur_mean, (cur_sigma, _, __) = emp_result[0]

                im_values.append(cur_mean)
                im_sigmas.append(cur_sigma)
            return pd.DataFrame(
                index=feature_values,
                columns=["mu", "sigma"],
                data=np.asarray([im_values, im_sigmas]).T,
            )

        elif feature_key == "mag":
            site = classdef.Site(
                rrup=const_values.rrup,
                rjb=const_values.rjb,
                rx=const_values.rx,
                hw=True,
                rtvz=0,
                vs30=const_values.vs30,
                vs30measured=False,
            )

            for cur_mag in feature_values:
                cur_fault = classdef.Fault(
                    Mw=cur_mag,
                    rake=const_values.rake,
                    dip=const_values.dip,
                    ztor=const_values.ztor,
                )

                emp_result = emp_factory.compute_gmm(
                    cur_fault, site, classdef.GMM.Br_10, im, [period]
                )
                if period == 0:
                    cur_mean, (cur_sigma, _, __) = emp_result
                else:
                    cur_mean, (cur_sigma, _, __) = emp_result[0]

                im_values.append(cur_mean)
                im_sigmas.append(cur_sigma)
            return pd.DataFrame(
                index=feature_values,
                columns=["mu", "sigma"],
                data=np.asarray([im_values, im_sigmas]).T,
            )

        elif feature_key == "vs30":
            fault = classdef.Fault(
                Mw=const_values.mag,
                rake=const_values.rake,
                dip=const_values.dip,
                ztor=const_values.ztor,
            )

            for cur_vs30 in feature_values:
                cur_site = classdef.Site(
                    rrup=const_values.rrup,
                    rjb=const_values.rjb,
                    rx=const_values.rx,
                    hw=True,
                    rtvz=0,
                    vs30=cur_vs30,
                    vs30measured=True,  # not sure about this parameter
                )
                emp_result = emp_factory.compute_gmm(
                    fault, cur_site, classdef.GMM.Br_10, im, [period]
                )
                if period == 0:
                    cur_mean, (cur_sigma, _, __) = emp_result
                else:
                    cur_mean, (cur_sigma, _, __) = emp_result[0]

                im_values.append(cur_mean)
                im_sigmas.append(cur_sigma)
            return pd.DataFrame(
                index=feature_values,
                columns=["mu", "sigma"],
                data=np.asarray([im_values, im_sigmas]).T,
            )

    def get_est_site_values(
        self,
        feature_key: str,
        feature_values: np.ndarray,
        const_values: pd.Series,
        locations: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Computes the estimated mean & std using the
        given NN model, varied along one feature at each of the
        given locations

        Parameters
        ----------
        feature_key: str, key
            The feature to vary
        feature_values: numpy array of floats
            The feature values at which to compute the IM value
        const_values: series
            The constant feature values to use
        locations: dataframe
            Locations of interest,
            required columsn [lon, lat]

        Returns
        -------
        feature_df: dataframe
            The feature dataframe used
        mean_est_df: dataframe
        std_est_df: dataframe
            The mean and std datadrames estimations
        """
        if locations is not None:
            feature_df = pd.DataFrame(
                data=np.concatenate(
                    (
                        np.repeat(locations.values, len(feature_values), axis=0),
                        np.tile(feature_values, locations.shape[0])[:, np.newaxis],
                    ),
                    axis=1,
                ),
                columns=list(locations.columns) + [feature_key],
            )
        else:
            feature_df = pd.DataFrame(data=feature_values, columns=[feature_key])

        constants_df = pd.DataFrame(
            data=np.repeat(
                const_values.values[np.newaxis, :], feature_df.shape[0], axis=0
            ),
            columns=const_values.index.values,
        )

        # Drop the column that is varied
        constants_df.drop(columns=[feature_key], inplace=True)

        # Merge
        feature_df = feature_df.merge(constants_df, left_index=True, right_index=True)

        # Run estimation
        mean_est_df, std_est_df = self.model.predict(feature_df, pre_process=True)
        mean_est_df[["lon", "lat", feature_key]] = feature_df[
            ["lon", "lat", feature_key]
        ]
        std_est_df[["lon", "lat", feature_key]] = feature_df[
            ["lon", "lat", feature_key]
        ]
        return feature_df, (mean_est_df, std_est_df)

    def gen_plot(
        self,
        im: str,
        feature_key: str,
        feature_df: pd.DataFrame,
        mean_est_df: pd.DataFrame,
        locations: pd.DataFrame = None,
        emp_df: pd.Series = None,
        output_ffp: Union[str, Path] = None,
    ):
        """Generates an IM vs. feature plot

        Parameters
        ----------
        im: str
            IM of interest
        feature_key: str
            The feature that was varied
        feature_df: dataframe
            The feature values that were used for estimation
        mean_est_df: dataframe
            The estimated mean values
        locations: dataframe, optional
            The locations at which the model was evaluated
        emp_df: dataframe, optional

        output_ffp:

        Returns
        -------

        """
        output_ffp = output_ffp if isinstance(output_ffp, Path) else Path(output_ffp)

        # Create the plot
        # fig = plt.figure(figsize=(18, 13.5))
        fig = plt.figure(figsize=(7, 5.5))

        if locations is None:
            plt.plot(feature_df[feature_key], mean_est_df[im], marker="x")
        else:
            if locations.shape[0] < len(MARKERS):
                for loc_ix in range(locations.shape[0]):
                    cur_loc_mask = (feature_df["lon"] == locations.iloc[loc_ix].lon) & (
                        feature_df["lat"] == locations.iloc[loc_ix].lat
                    )
                    plt.scatter(
                        feature_df.loc[cur_loc_mask, feature_key],
                        mean_est_df.loc[cur_loc_mask, im],
                        marker=MARKERS[loc_ix],
                    )
            else:
                plt.scatter(feature_df[feature_key], mean_est_df[im], marker=".", s=1)

            # Add mean & std line
            feature_means = mean_est_df.groupby(feature_key).mean()
            feature_stds = mean_est_df.groupby(feature_key).std()
            plt.plot(
                feature_means.index.values,
                feature_means[im],
                linewidth=0.75,
                c="k",
                label="Mean of Sites - NN",
            )
            plt.plot(
                feature_means.index.values,
                feature_means[im] + feature_stds[im],
                linestyle="--",
                c="k",
                linewidth=0.75,
                label="Std of Sites - NN",
            )
            plt.plot(
                feature_means.index.values,
                feature_means[im] - feature_stds[im],
                linestyle="--",
                c="k",
                linewidth=0.75,
            )

        if emp_df is not None:
            add_emp(emp_df)

        plt.xlabel(feature_key)
        plt.ylabel(im)
        plt.yscale("log")
        plt.grid(linestyle="--", linewidth=0.25)
        plt.legend()
        plt.title("{} {}".format(im, feature_key))

        if output_ffp is not None:
            if not output_ffp.parent.is_dir():
                output_ffp.parent.mkdir(parents=True)
            plt.savefig(output_ffp, dpi=300)
            plt.close()
        else:
            plt.show()

        return fig

    def gen_mean_plot(
        self,
        im: str,
        feature_key: str,
        feature_df: pd.DataFrame,
        mean_est_df: pd.DataFrame,
        std_est_df: pd.DataFrame,
        emp_df: pd.Series = None,
        output_ffp: Union[str, Path] = None,
    ):
        """Very similar to gen_plot, except that it takes the
        mean of the means & stds at each location and plots those

        This is not ideal, mainly exists for the presentation
        """
        # fig = plt.figure(figsize=(18, 13.5))
        fig = plt.figure(figsize=(7, 5.5))

        x = feature_df[feature_key].unique()
        mean = mean_est_df.groupby(feature_key).mean()[im]
        std = std_est_df.groupby(feature_key).mean()[im]
        plt.plot(x, mean, c="k", linewidth=0.75, label="Mean NN")
        plt.plot(
            x, mean * np.exp(std), c="k", linewidth=0.75, linestyle="--", label="Std NN"
        )
        plt.plot(x, mean * np.exp(-std), c="k", linewidth=0.75, linestyle="--")

        # Empirical plots
        if emp_df is not None:
            add_emp(emp_df)

        plt.xlabel(feature_key)
        plt.ylabel(im)
        plt.yscale("log")
        plt.grid(linestyle="--", linewidth=0.25)
        plt.legend()
        plt.title("{} {}".format(im, feature_key))

        if output_ffp is not None:
            if not output_ffp.parent.is_dir():
                output_ffp.parent.mkdir(parents=True)
            plt.savefig(output_ffp, dpi=300)
            plt.close()
        else:
            plt.show()

        return fig

    def gen_plots(
        self,
        ims: np.ndarray,
        feature_dict: Dict[str, np.ndarray],
        output_dir: Union[Path, None],
        locations: pd.DataFrame = None,
        mean: bool = False,
    ):
        """Generates plots for each IM

        Parameters
        ----------
        See gen_plot
        """
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)

        for cur_feature, cur_feature_values in feature_dict.items():
            (
                cur_feature_df,
                (cur_mean_est_df, cur_std_est_df,),
            ) = self.get_est_site_values(
                cur_feature, cur_feature_values, CONST_DEFAULT_VALUES, locations
            )

            for cur_im in ims:
                cur_emp_df = self.get_B13_values(
                    cur_im, cur_feature, cur_feature_values, self.CONST_DEFAULT_VALUES
                )

                if not mean:
                    self.gen_plot(
                        cur_im,
                        cur_feature,
                        cur_feature_df,
                        cur_mean_est_df,
                        locations,
                        emp_df=cur_emp_df,
                        output_ffp=output_dir
                        / f"{cur_im.replace('.', 'p')}_{cur_feature}.png",
                    )
                else:
                    self.gen_mean_plot(
                        cur_im,
                        cur_feature,
                        cur_feature_df,
                        cur_mean_est_df,
                        cur_std_est_df,
                        emp_df=cur_emp_df,
                        output_ffp=output_dir
                        / f"{cur_im.replace('.', 'p')}_{cur_feature}.png",
                    )


class PlotGen:
    def __init__(
        self, plot_items_ffp: str, model: GMM, data_dirs: List[Path], output_dir: Path
    ):
        self.plot_items_ffp = plot_items_ffp

        self.model = model

        self.data_dirs = data_dirs
        self.output_dir = output_dir

        self._estimates = {}

    def _get_event_estimates(
        self, event: str,
    ):
        """Get estimates for the specified event"""
        if event in self._estimates.keys():
            return self._estimates[event]
        else:
            # Find the .tfrecord file
            record_ffp, feature_details = find_record_ffp(self.data_dirs, event)

            # Load the data
            df = data.load_tfrecord(str(record_ffp), feature_details)

            # Get the estimates
            mean_est, std_est = self.model.predict(df.loc[:, self.model.features])

            self._estimates[event] = (df, mean_est, std_est)
            return df, mean_est, std_est


class AggPlotGen(PlotGen):
    def __init__(
        self,
        plot_items_ffp: str,
        model: GMM,
        ims: List[str],
        data_dirs: List[Path],
        output_dir: Path,
    ):
        super().__init__(plot_items_ffp, model, data_dirs, output_dir)
        self._ims = ims

        self._sim_df, self._mean_df, self._std_df = self._get_estimates()

        self._sim_df["station"] = get_station_from_id(
            self._sim_df.index.values.astype(str)
        )
        self._mean_df["station"] = get_station_from_id(
            self._mean_df.index.values.astype(str)
        )
        self._std_df["station"] = get_station_from_id(
            self._std_df.index.values.astype(str)
        )

        if not self.output_dir.is_dir():
            self.output_dir.mkdir()

    def _get_estimates(self):
        return self.model.predict_dirs(
            self.data_dirs, pre_process=True, ims=self._ims, features=["lat", "lon"]
        )

    def _get_event_estimates(self, event: str):
        raise NotImplementedError()

    def plot_spatial_res_maps(self, plot_type: str = "res_mean", n_procs: int = 4):
        assert np.all(self._sim_df.index == self._mean_df.index)

        data_df, non_negative = None, True
        if plot_type == "res_mean":
            data_df = (
                self._mean_df[self._ims] / self._sim_df[self._ims].apply(np.exp)
            ).apply(np.log)

        data_df["station"] = self._mean_df.station
        data_df = data_df.groupby("station").mean()

        # Add lat & lon
        station_df = get_station_lookup(self._sim_df)
        data_df = pd.merge(
            data_df,
            station_df,
            left_on="station",
            right_index=True,
            how="inner",
            validate="one_to_one",
        )

        csv_files = []
        for im in self._ims:
            im_name = im.replace(".", "p")
            cb_options = compute_GMT_std_ticks(
                data_df[im], n_std=2, non_negative=non_negative
            )
            gmt_options = get_gmt_options_dict(
                options={
                    **{"title": f"{im_name}-{plot_type}", "xyz-cpt-labels": f"{im}",},
                    **cb_options,
                }
            )

            plot_csv_ffp = self.output_dir / f"{im_name}"
            plot_csv_ffp = _gmt_save(
                data_df, im, str(plot_csv_ffp), gmt_options=gmt_options
            )

            csv_files.append(plot_csv_ffp)

        # Generate the plot
        plot_multiple(
            self.plot_items_ffp,
            DEFAULT_STANDARD_GMT_PLOT_OPTIONS,
            in_ffps=csv_files,
            n_procs=n_procs
        )

        return csv_files


class EventPlotGen(PlotGen):
    """Class for generating even specific plots"""

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
        std_df[feature_key] = sim_df[feature_key]

        bin_edges = np.linspace(x.min() - 1e-5, x.max(), 50)
        bin_indices = np.digitize(sim_df[feature_key].values, bin_edges, right=True)

        # Get center points of the bins (there might be a better way of doing this
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        mean_df["bin_ix"], std_df["bin_ix"] = bin_indices, bin_indices
        bin_mean_mean = mean_df.groupby("bin_ix").mean()[im]
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
        plt.grid(linestyle="--", linewidth=0.25)
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
        cb_options: Dict[str, Dict] = None,
        suffix: str = "",
        n_procs: int = 4,
    ):
        """Plots spatial maps for the specified IMs and events
        See gen_event_map_data for parameter details
        """
        for event in events:
            self.plot_spatial_event_maps(
                event,
                ims,
                data_type=data_type,
                cb_options=cb_options,
                suffix=suffix,
                n_procs=n_procs,
            )

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
        plot_csv_ffp = self.gen_event_map_data(
            event, im, data_type=data_type, suffix=suffix
        )

        # Generate the plot
        plot_multiple(
            self.plot_items_ffp,
            DEFAULT_STANDARD_GMT_PLOT_OPTIONS,
            in_ffps=[str(plot_csv_ffp)],
        )

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
            GMT colourbar options, if not given then
            these will be calculated based on the current data
            Useful when wanting two plots with the
            same colourbar scale
        suffix: str, optional
            Filename suffix

        Returns
        -------
        str
            Path to the data csv
        dict
            The colourbar options used
        """
        event_out_dir = self.output_dir / event
        if not event_out_dir.is_dir():
            event_out_dir.mkdir(parents=True)

        sim_df, mean_est, std_est = self._get_event_estimates(event)
        assert np.all(mean_est.index.values == sim_df.index.values)

        non_negative = True
        if data_type.lower() == "sim":
            data_df = sim_df[im].apply(np.exp).to_frame()
        elif data_type.lower() == "est_mean":
            data_df = mean_est.copy()
        elif data_type.lower() == "est_std":
            data_df = std_est.copy()
        elif data_type.lower() == "res_mean":
            data_df = (mean_est[im] / sim_df[im].apply(np.exp)).apply(np.log).to_frame()
            non_negative, cb_options = False, {im: {}}
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        data_df = pd.merge(
            data_df,
            sim_df.loc[:, ["lat", "lon"]],
            left_index=True,
            right_index=True,
            how="inner",
        )

        assert data_df.shape[0] == mean_est.shape[0]

        data_df["station"] = get_station_from_id(data_df.index.values.astype(str))

        # Aggregate across realisation at each station
        agg_df = data_df.groupby("station").mean()

        im_name = im.replace(".", "p")
        cb_options = (
            compute_GMT_std_ticks(agg_df[im], n_std=2, non_negative=non_negative)
            if cb_options is None
            else cb_options[im]
        )
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


def find_record_ffp(data_dirs: List[Path], event: str):
    """Finds the tfrecord file for the given event in
    the specified directories, raises ValueError if no
    file is found
    """
    results = []
    for cur_dir in data_dirs:
        cur_r = list(cur_dir.glob(f"{event}.tfrecord"))

        if len(cur_r) > 0:
            cur_feature_details = data.load_feature_details(cur_dir)
            results.append((cur_r[0], cur_feature_details))

    if len(results) == 0:
        raise ValueError(
            f"No tfrecord file could be found for the specified event: {event}"
        )

    assert len(results) == 1, "More than one tfrecord file found"

    return results[0]


def _gmt_save(df: pd.DataFrame, key: str, output_ffp: str, gmt_options: Dict = None):
    """Saves the specified data in the correct csv format for GMT plotting,
    also creates the corresponding options file

    Parameters
    ----------
    df: dataframe
        Contains the relevant data, has to
        have columns [lon, lat, key]
    key: str
        Column name that contains the values to save
    output_ffp: str
        Output file path without the extension
    gmt_options: dictionary, optional
        The individual GMT options dict

    Returns
    ----------
    string
        Name of the output csv file
    """
    df.loc[:, ["lon", "lat", key]].rename(columns={key: "value"}).to_csv(
        f"{output_ffp}.csv"
    )

    gmt_options = get_gmt_options_dict() if gmt_options is None else gmt_options
    with open(f"{output_ffp}.yaml", "w") as f:
        yaml.safe_dump(gmt_options, f)

    return f"{output_ffp}.csv"


def get_gmt_options_dict(flags: List[str] = None, options: Dict[str, Any] = None):
    """Generates the GMT plot options dict"""
    cur_dict = TEMPLATE_OPTIONS_DICT.copy()

    if flags is not None:
        cur_dict["flags"] = flags

    if options is not None:
        cur_dict["options"] = options

    return cur_dict


def compute_GMT_std_ticks(
    data_series: pd.Series,
    n_std: float = 2,
    center: float = None,
    non_negative: bool = True,
):
    options = {}

    std = data_series.std()
    center = center if center is not None else data_series.mean()

    # Compute min/max and tick increments for colour bar
    cpt_max = center + (n_std * std)
    n_dec_points = -int(f"{std:.10E}".split("E")[1])

    # Can't use the np functions, as yaml does not know
    # how to save type np.float
    cpt_max = round(float(cpt_max), n_dec_points)
    center = round(float(center), n_dec_points)

    tick_inc = float(
        round(((cpt_max - center + (1 * 10 ** (-n_dec_points - 4))) / 4), n_dec_points)
    )
    cpt_max = round(center + (tick_inc * 4), n_dec_points)
    cpt_min = round(center - (4 * tick_inc), n_dec_points)

    options["xyz-cpt-max"] = cpt_max
    options["xyz-cpt-min"] = max(0, cpt_min) if non_negative else cpt_min
    options["xyz-cpt-tick"], options["xyz-cpt-inc"] = tick_inc, tick_inc / 2

    return options


def add_emp(emp_df: pd.Series, label: str = "Bradley 2013"):
    """Adds the empirical data to the
    current plot

    Parameters
    ----------
    emp_df: dataframe
        The empirical GMM estimates, expects
        the columns [mu, sigma]
    """
    # mean prediction
    plt.plot(
        emp_df.index.values, emp_df["mu"].values, c="r", label="Mean Bradley 2013."
    )
    # +- sigma
    plt.plot(
        emp_df.index.values,
        emp_df["mu"].values * np.exp(emp_df["sigma"].values),
        c="r",
        linestyle="--",
        label=f"Std {label}",
    )
    plt.plot(
        emp_df.index.values,
        emp_df["mu"].values * np.exp(-emp_df["sigma"].values),
        c="r",
        linestyle="--",
        label=f"Mean {label}",
    )


def set_plot_lims(
    x: Union[np.ndarray, List[np.ndarray]], y: Union[np.ndarray, List[np.ndarray]]
):
    x = x if isinstance(x, list) else [x]
    y = y if isinstance(y, list) else [y]

    x_min = np.min([cur_x.min() for cur_x in x])
    y_min = np.min([cur_y.min() for cur_y in y])

    x_max = np.max([cur_x.max() for cur_x in x])
    y_max = np.max([cur_y.max() for cur_y in y])

    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
