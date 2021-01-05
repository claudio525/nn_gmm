from typing import Tuple, Iterable, Callable, Dict, List, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from visualization.gmt.plotting import plot_multiple, plot_single

from .. import utils
from .. import GMM
from .. import eval
from . import plotting_utils as plt_utils


class AggPlotGen(plt_utils.ModelEventBasePlotGen):
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

        self._sim_df, self._mean_df, self._std_df = None, None, None

        self._rel_res_df = None

        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True)

    @property
    def rel_res_df(self):
        if self._rel_res_df is None:
            self._rel_res_df = eval.get_realisation_residuals(
                self.data_dirs, self.model
            )
        return self._rel_res_df

    @property
    def sim_df(self):
        if self._sim_df is None:
            self._get_estimates()
        return self._sim_df

    @property
    def mean_df(self):
        if self._sim_df is None:
            self._get_estimates()
        return self._mean_df

    @property
    def std_df(self):
        if self._sim_df is None:
            self._get_estimates()
        return self._std_df

    def _get_estimates(self):
        print(f"Getting estimates for the specified IMs")
        self._sim_df, self._mean_df, self._std_df = self.model.predict_dirs(
            self.data_dirs, ims=self._ims, features=["lat", "lon", "mag"]
        )

        print("Adding station data")
        self._sim_df["station"] = utils.get_station_from_id(
            self._sim_df.index.values.astype(str)
        )
        self._mean_df["station"] = self._sim_df.station.values

        if self._std_df is not None:
            self._std_df["station"] = self._sim_df.station.values

    def _get_event_estimates(self, event: str):
        raise NotImplementedError()

    def plot_realisation_residuals(self):
        """Creates a residual plot for realisations"""

        print("Plotting realisation residuals")
        for im in self._ims:

            fig = px.scatter(
                data_frame=self.rel_res_df,
                x="mag",
                y=im,
                # symbol="fault",
                # symbol_sequence=list(range(45)),
                hover_name=np.char.add(
                    np.char.add(self.rel_res_df.index.values.astype(str), " - "),
                    self.rel_res_df.loc[:, "n_stations"].values.astype(str),
                ),
                title=f"{im} - Realisation ln residuals (mean)",
                labels={
                    "x": f"{im} - Realisation ln residual (mean)",
                    "y": f"Mean ln abs residual",
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
                str(self.output_dir / f"{im}_realisation_ln_abs_residual.html")
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
        station_df = utils.get_station_lookup(self._sim_df)
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
                else plt_utils.compute_GMT_std_ticks(
                    data_df[im], n_std=2, non_negative=non_negative
                )
            )
            gmt_options = plt_utils.get_gmt_options_dict(
                options={
                    **{"title": f"{im_name}-{plot_type}", "xyz-cpt-labels": f"{im}",},
                    **cb_options,
                }
            )

            plot_csv_ffp = self.output_dir / f"{im_name}"
            plot_csv_ffp = plt_utils.gmt_save(
                data_df, im, str(plot_csv_ffp), gmt_options=gmt_options
            )

            csv_files.append(plot_csv_ffp)

        # Generate the plot
        print(f"Plotting")
        plot_multiple(
            self.plot_items_ffp,
            plt_utils.DEFAULT_RES_GEN_GMT_PLOT_OPTIONS
            if "res" in plot_type
            else plt_utils.DEFAULT_STANDARD_GMT_PLOT_OPTIONS,
            in_ffps=csv_files,
            n_procs=n_procs,
        )

        return csv_files
