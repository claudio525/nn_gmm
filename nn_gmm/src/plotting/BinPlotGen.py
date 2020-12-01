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


class BinPlotGen(ModelEventBasePlotGen):
    """Creates plots for the binned dataset"""

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
        self.vs30_bins = vs30_bins if vs30_bins is not None else DEFAULT_VS30_BINS

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
            self.output_dir / f"{dataset}_{im.replace('.', 'p')}.png",
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
            f"{dataset}_{im.replace('.', 'p')}_binned_residual_plots.png"
            if not log_space
            else f"{dataset}_{im.replace('.', 'p')}_binned_log_ratio_plots.png"
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