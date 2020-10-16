import gc
import tempfile
from typing import Tuple, Iterable, Callable, Dict, List, Any, Union
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import empirical.util.classdef as classdef
import empirical.util.empirical_factory as emp_factory
from visualization.gmt.plotting import plot_multiple, plot_single

from .model import GMM
from .utils import get_station_from_id, get_station_lookup, to_path, to_list
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

PLOT_TYPE_OPTIONS_MAPPING = {
    "sim": DEFAULT_STANDARD_GMT_PLOT_OPTIONS,
    "est_mean": DEFAULT_STANDARD_GMT_PLOT_OPTIONS,
    "est_std": DEFAULT_RES_GEN_GMT_PLOT_OPTIONS,
    "res_mean": DEFAULT_RES_GEN_GMT_PLOT_OPTIONS,
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
            "strike": 177,
            "length": 18.9,
            "width": 18.2775,
            "hdepth": 8.990423e00,
            "ztor": 0,
            "dbottom": 16.8,
            "is_point_source": 0,
            "mag": 7.0,
            "rjb": 91,
            "rrup": 91,
            "rx": 91,
            "ry": 91,
            "s": 2.760520e01,
            "theta": 4.517924e01,
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
        if std_est_df is not None:
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
                cur_feature, cur_feature_values, self.CONST_DEFAULT_VALUES, locations
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
    """Class for generating aggregate plots"""

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

        print(f"Getting estimates for the specified IMs")
        self._sim_df, self._mean_df, self._std_df = self._get_estimates()
        assert np.all(self._sim_df.index == self._mean_df.index)

        print("Adding station data")
        self._sim_df["station"] = get_station_from_id(
            self._sim_df.index.values.astype(str)
        )
        self._mean_df["station"] = self._sim_df.station.values

        if self._std_df is not None:
            self._std_df["station"] = self._sim_df.station.values

        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True)

    def _get_estimates(self):
        return self.model.predict_dirs(
            self.data_dirs, ims=self._ims, features=["lat", "lon", "mag"]
        )

    def _get_event_estimates(self, event: str):
        raise NotImplementedError()

    def plot_realisation_residuals(self, abs_residual: bool = False):
        """Creates a residual plot for realisations"""
        print("Plotting realisation residuals")
        assert np.all(self._mean_df.index == self._sim_df.index)

        split_ids = np.stack(np.char.split(self._mean_df.index.values.astype(str), "_"))
        self._mean_df["realisation"] = np.char.add(
            np.char.add(split_ids[:, 0], "_"), split_ids[:, 1]
        )
        self._mean_df["fault"] = split_ids[:, 0]

        self._sim_df["realisation"] = self._mean_df.realisation

        rel_fault = self._mean_df.groupby("realisation").first()["fault"].to_frame()

        rel_mag = self._sim_df.groupby("realisation").first()["mag"]
        rel_n_stations = self._sim_df.groupby("realisation").count()["station"]

        for im in self._ims:
            res = (
                self._sim_df.loc[:, im] - self._mean_df.loc[:, im].apply(np.log)
            ).to_frame()
            if abs_residual:
                res[im] = res[im].apply(np.abs)

            res["realisation"] = self._mean_df.realisation

            rel_res = res.groupby("realisation").mean()
            rel_res = rel_res.merge(rel_mag, left_index=True, right_index=True)
            rel_res = rel_res.merge(rel_fault, left_index=True, right_index=True)

            assert np.all(rel_mag.index == rel_res.index)

            fig = px.scatter(
                data_frame=rel_res,
                x="mag",
                y=im,
                symbol="fault",
                symbol_sequence=list(range(45)),
                hover_name=np.char.add(
                    np.char.add(rel_res.index.values.astype(str), " - "),
                    rel_n_stations.values.astype(str),
                ),
                title=f"{im} - Realisation ln residuals (mean)",
                labels={
                    "x": im,
                    "y": f"Mean ln {'abs' if abs_residual else ''} residual",
                },
                marginal_y="histogram",
            )
            fig.update_layout(
                shapes=[
                    {
                        "type": "line",
                        "xref": "paper",
                        "y0": 0,
                        "y1": 0,
                        "yref": "y",
                        "x0": 0,
                        "x1": 1,
                    }
                ],
                showlegend=False,
            )
            fig.write_html(
                str(
                    self.output_dir
                    / f"{im}_realisation_ln_{'abs_' if abs_residual else ''}residual.html"
                )
            )

        return

    def plot_spatial_agg_maps(self, plot_type: str = "res_mean", n_procs: int = 4):
        """Generates spatial aggregate maps (i.e. the data is aggregated
        at each station across all realisations of all events

        Parameters
        ----------
        plot_type: str
            "res_mean": Residual between simulation IMs and
            estimated mean value from the NN GMM
        n_procs: int, optional
            Number of processes to use for plotting

        Returns
        -------
        list of strings:
            The csv file paths for each plot
        """
        is_res_plot = "res" in plot_type

        data_df, non_negative = None, True
        if plot_type == "res_mean":
            data_df = (
                self._mean_df[self._ims] / self._sim_df[self._ims].apply(np.exp)
            ).apply(np.log)
            non_negative = False

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

        print(f"Generating csv files for {len(self._ims)} IMs")
        csv_files = []
        for im in self._ims:
            im_name = im.replace(".", "p")
            cb_options = (
                {}
                if is_res_plot
                else compute_GMT_std_ticks(
                    data_df[im], n_std=2, non_negative=non_negative
                )
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
        print(f"Plotting")
        plot_multiple(
            self.plot_items_ffp,
            DEFAULT_RES_GEN_GMT_PLOT_OPTIONS
            if "res" in plot_type
            else DEFAULT_STANDARD_GMT_PLOT_OPTIONS,
            in_ffps=csv_files,
            n_procs=n_procs,
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


class BinPlotGen(PlotGen):
    """Creates plots for the binned dataset"""

    DEFAULT_VS30_BINS = [0, 200, 400, 600, 800, 1200]

    def __init__(
        self,
        plot_items_ffp: str,
        model: GMM,
        data_sets: Dict[str, Path],
        output_dir: Path,
        mag_bins: List[float] = None,
        vs30_bins: List[float] = None,
    ):
        super().__init__(plot_items_ffp, model, list(data_sets.values()), output_dir)

        self.mag_bins = mag_bins
        self.vs30_bins = vs30_bins if vs30_bins is not None else self.DEFAULT_VS30_BINS

        self.data_sets = data_sets

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    def create_IM_bin_plot(self, dataset: str, im: str):
        data_set_dir = self.data_sets[dataset]

        # Get the events of the dataset
        events = [
            event_record.name.split(".")[0]
            for event_record in data_set_dir.glob("*.tfrecord")
        ]

        # Get the required data
        data_dfs, mean_est_dfs = [], []
        for event in events:
            df, mean_est, std_est = self._get_event_estimates(event)
            data_dfs.append(df.loc[:, ["mag", "vs30", "rrup", im]])
            mean_est_dfs.append(mean_est.loc[:, im])

        df, mean_est = pd.concat(data_dfs), pd.concat(mean_est_dfs)

        mag_bins = (
            self.mag_bins
            if self.mag_bins is not None
            else self._get_default_mag_bins(df.mag.values)
        )

        plot_mag_vs30_bins(
            df,
            mean_est,
            im,
            np.asarray(mag_bins),
            np.asarray(self.vs30_bins),
            self.output_dir / f"{dataset}_{im}.png",
        )

    def create_IM_res_scatter_bin_plot(
        self, dataset: str, im: str, log_space: bool = False
    ):
        """Creates an residual scatter plot with x- & y-axis histograms
        for each magnitude and vs30 bin
        """
        data_set_dir = self.data_sets[dataset]

        # Get the events of the dataset
        events = [
            event_record.name.split(".")[0]
            for event_record in data_set_dir.glob("*.tfrecord")
        ]

        # Get the required data
        data_dfs, mean_est_dfs = [], []
        for event in events:
            df, mean_est, std_est = self._get_event_estimates(event)
            data_dfs.append(df.loc[:, ["mag", "vs30", "rrup", im]])
            mean_est_dfs.append(mean_est.loc[:, im])

        df, mean_est = pd.concat(data_dfs), pd.concat(mean_est_dfs)
        assert np.all(df.index == mean_est.index)

        mag_bins = (
            self.mag_bins
            if self.mag_bins is not None
            else self._get_default_mag_bins(df.mag.values)
        )

        filename = (
            f"{dataset}_{im}_binned_residual_plots.png"
            if not log_space
            else f"{dataset}_{im}_binned_log_ratio_plots.png"
        )

        plot_mag_vs30_res_bins(
            df,
            mean_est,
            im,
            mag_bins,
            np.asarray(self.vs30_bins),
            self.output_dir / filename,
            log_space=log_space,
        )

    def _get_default_mag_bins(self, mag_values: np.ndarray):
        return np.arange(np.floor(np.min(mag_values)), np.ceil(np.max(mag_values)) + 1)


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
    fig = multi_fig((8, 6), n_rows, n_cols)
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

            residual_hist_plot(
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


def plot_mag_vs30_bins(
    df: pd.DataFrame,
    mean_est_df: pd.DataFrame,
    im: str,
    mag_bins: np.ndarray,
    vs30_bins: np.ndarray,
    output_ffp: Path,
):
    """Creates IM value vs rrup scatter plots for each of the
    magnitude and vs30 bins
    """
    mag_ind = np.digitize(df.mag.values, mag_bins)
    vs_30_ind = np.digitize(df.vs30.values, vs30_bins)

    n_rows, n_cols = len(mag_bins) - 1, len(vs30_bins) - 1
    fig = multi_fig((8, 6), n_rows, n_cols)
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

            cur_ax.scatter(df.rrup[cur_mask], np.exp(df.loc[cur_mask, im]), s=1.0)
            cur_ax.scatter(df.rrup[cur_mask], mean_est_df.loc[cur_mask], s=1.0)
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


def plot_n_records_map(
    data_dirs: List[Union[Path, str]],
    plot_items_ffp: Union[Path, str],
    output_ffp: str,
    title: str = "Number-of-records",
):
    """Generates a spatial map that shows number of records at each station"""
    data_dirs, plot_items_ffp = to_path(data_dirs), to_path(plot_items_ffp)

    ds = data.load_dataset(
        data_dirs,
        data.load_feature_details(data_dirs[0]),
        5_000_000,
        shuffle_buffer=None,
        block_size=1024,
    )

    dfs = []
    for cur_batch in ds.as_numpy_iterator():
        cur_df = pd.DataFrame.from_dict(
            {key: cur_batch[key] for key in ["id", "lat", "lon"]}
        )
        cur_df["id"] = cur_df.id.str.decode("UTF-8")
        cur_df.set_index("id", inplace=True)

        dfs.append(cur_df)

    df = pd.concat(dfs)
    df["station"] = get_station_from_id(df.index.values.astype(str))

    station_lookup_df = get_station_lookup(df)

    n_records_df = df.groupby("station").count()
    n_records_df["count"] = n_records_df["lat"]
    n_records_df.drop(columns=["lat", "lon"], inplace=True)

    n_records_df = pd.merge(
        n_records_df, station_lookup_df, how="inner", right_index=True, left_index=True
    )

    cb_options = compute_GMT_std_ticks(n_records_df["count"], non_negative=False)
    gmt_options = get_gmt_options_dict(
        options={**{"title": title, "xyz-cpt-labels": "n_records"}, **cb_options,}
    )
    csv_ffp = _gmt_save(n_records_df, "count", output_ffp, gmt_options=gmt_options)

    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_single(plot_items_ffp, csv_ffp, DEFAULT_STANDARD_GMT_PLOT_OPTIONS, tmp_dir)

    return


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


def residual_hist_plot(
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
