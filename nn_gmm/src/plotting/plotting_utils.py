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

IM_MEAN_KEY, IM_STD_KEY = "{}_mean", "{}_std"
TEMPLATE_OPTIONS_DICT = {"flags": [], "options": {}}

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

DEFAULT_VS30_BINS = [0, 200, 400, 600, 800, 1200]

class ModelEventBasePlotGen:
    """Base class for generating plots that use NN-GMM model
    predictions for an event from the training or validation dataset"""

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
