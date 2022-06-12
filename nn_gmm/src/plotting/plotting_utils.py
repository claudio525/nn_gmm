from typing import Tuple, Iterable, Callable, Dict, List, Any, Union
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nn_gmm.src.model import NeuralNetworkGMM
from nn_gmm.src import utils
from nn_gmm.src import data

FEATUER_NAME_LOOKUP = {
    "rrup": r"$R_{Rup}$",
    "vs30": r"$V_{S30}$",
    "mag": r"$M_w$"
}

IM_NAME_LOOKUP = {}

DEFAULT_VS30_BINS = [0, 200, 400, 600, 800, 1200]


DEFAULT_PLOT_KWARGS = dict(linewidth=0.75)

def get_feature_name(id: str):
    """Gets the proper feature name for the specified 'id'"""
    name = FEATUER_NAME_LOOKUP.get(id)

    if name is None:
        return id
    return name


def get_im_name(id: str):
    """Gets the proper feature name for the specified 'id'"""
    name = IM_NAME_LOOKUP.get(id)

    if id.startswith("pSA"):
        return f"pSA({id.split('_')[-1]}s)"

    if name is None:
        return id
    return name


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


def gmt_save(df: pd.DataFrame, key: str, output_ffp: str, gmt_options: Dict = None):
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
        f"{output_ffp}.csv", index=False
    )

    with open(f"{output_ffp}.yaml", "w") as f:
        yaml.safe_dump(gmt_options, f)

    return f"{output_ffp}.csv"


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
