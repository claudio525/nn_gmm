from typing import List, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from nn_gmm.src.console import console
from nn_gmm.src.ResultDB import ResultDB
from . import plotting_funcs as pf
from . import plotting_utils as pu


class BinPlotGen:
    """Creates multi-plot figures with different
    data bins (split based on vs30 & magnitude)"""

    def __init__(
        self, mag_bins: List[float] = None, vs30_bins: List[float] = None,
    ):
        self.mag_bins = mag_bins
        self.vs30_bins = vs30_bins if vs30_bins is not None else pu.DEFAULT_VS30_BINS

    def create_IM_bin_plot(
        self,
        im: str,
        output_dir: Path,
        result_db_ffp: Path,
        feature: str = "rrup",
        prefix: str = None,
    ):
        """Creates a multi-plot figure, with the
        specified feature on the x-axis"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get the data
        columns = [im, f"{im}_est", feature, "mag", "vs30"]
        data_df = ResultDB.get_data_static(result_db_ffp, columns)

        # Generate magnitude bins if not specified
        mag_bins = (
            self.mag_bins
            if self.mag_bins is not None
            else self._get_default_mag_bins(data_df.mag.values)
        )

        # Generate the plot
        prefix = f"{prefix}_" if prefix is not None else ""
        pf.plot_mag_vs30_bins(
            data_df,
            data_df[f"{im}_est"],
            im,
            np.asarray(mag_bins),
            np.asarray(self.vs30_bins),
            output_dir / f"{prefix}{feature}_{im.replace('.', 'p')}.png",
        )

    def _get_default_mag_bins(self, values: np.ndarray):
        return np.arange(np.floor(np.min(values)), np.ceil(np.max(values)) + 1)

    # def create_IM_res_scatter_bin_plot(
    #     self, dataset: str, im: str, log_space: bool = False
    # ):
    #     """Creates an residual scatter plot with x- & y-axis histograms
    #     for each magnitude and vs30 bin
    #     """
    #     data_set_dir = self.data_sets[dataset]
    #
    #     # Get the events of the dataset
    #     events = [
    #         event_record.name.split(".")[0]
    #         for event_record in data_set_dir.glob("*.tfrecord")
    #     ]
    #
    #     # Get the required data
    #     data_dfs, mean_est_dfs = [], []
    #     for event in events:
    #         df, mean_est, std_est = self._get_event_estimates(event)
    #         data_dfs.append(df.loc[:, ["mag", "vs30", "rrup", im]])
    #         mean_est_dfs.append(mean_est.loc[:, im])
    #
    #     df, mean_est = pd.concat(data_dfs), pd.concat(mean_est_dfs)
    #     assert np.all(df.index == mean_est.index)
    #
    #     mag_bins = (
    #         self.mag_bins
    #         if self.mag_bins is not None
    #         else self._get_default_mag_bins(df.mag.values)
    #     )
    #
    #     filename = (
    #         f"{dataset}_{im.replace('.', 'p')}_binned_residual_plots.png"
    #         if not log_space
    #         else f"{dataset}_{im.replace('.', 'p')}_binned_log_ratio_plots.png"
    #     )
    #
    #     pf.plot_mag_vs30_res_bins(
    #         df,
    #         mean_est,
    #         im,
    #         mag_bins,
    #         np.asarray(self.vs30_bins),
    #         self.output_dir / filename,
    #         log_space=log_space,
    #     )
    #
