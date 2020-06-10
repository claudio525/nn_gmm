import os
from typing import Tuple, Iterable, Callable, Dict, List, Any

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from visualization.gmt.plotting import plot_multiple
from . import evaluation

sns.set()

IM_MEAN_KEY, IM_STD_KEY = "{}_mean", "{}_std"
TEMPLATE_OPTIONS_DICT = {"flags": [], "options": {}}


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


def create_IM_res_hist(
    output_ffp: str,
    train_df: pd.DataFrame,
    im: str,
    val_df: pd.DataFrame = None,
    xlim_n_std: float = None,
):
    im_mean, im_std = f"{im}_mean", f"{im}_std"

    fig = plt.figure(figsize=(12, 6))
    ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    # Plot mean and std histogram
    for cur_key, ax in zip([im_mean, im_std], [ax1, ax2]):
        # Filter out data points that are not withing the specified limits
        # Added to prevent outliers extending x-axis to far
        if xlim_n_std is not None:
            min_x, max_x = __get_min_max_x(train_df.loc[:, cur_key].values, xlim_n_std)
            train_mask = (train_df[cur_key].values > min_x) & (
                train_df[cur_key].values < max_x
            )

            sns.distplot(train_df.loc[train_mask, cur_key], kde=False, ax=ax)
            if val_df is not None:
                val_mask = (val_df[cur_key].values > min_x) & (
                    val_df[cur_key].values < max_x
                )
                sns.distplot(val_df.loc[val_mask, cur_key], kde=False, ax=ax)
        else:
            sns.distplot(train_df[cur_key], kde=False, ax=ax)
            if val_df is not None:
                sns.distplot(val_df[cur_key], kde=False, ax=ax)

    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close()


def __get_min_max_x(data: np.ndarray, xlim_n_std: float):
    std_lim = np.nanstd(data) * xlim_n_std
    min_x = -std_lim if -std_lim > np.nanmin(data) else np.nanmin(data)
    max_x = std_lim if std_lim < np.nanmax(data) else np.nanmax(data)

    return min_x, max_x


def create_res_hist(
    output_ffp: str,
    train_df: pd.DataFrame,
    ims: Iterable[str],
    val_df: pd.DataFrame = None,
    xlim_n_std: float = None,
):
    """Creates a residual histogram plot across IMs, should probably
    be combined with create_IM_res_hist.."""
    fig = plt.figure(figsize=(12, 6))
    ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    for cur_temp, ax, cur_label in zip(
        [IM_MEAN_KEY, IM_STD_KEY],
        [ax1, ax2],
        [r"$ln \frac{\hat{\mu}}{\mu}$", r"$ln \frac{\hat{\sigma}}{\sigma}$"],
    ):
        cur_keys = [cur_temp.format(im) for im in ims]
        cur_train_data = train_df.loc[:, cur_keys].values.ravel()
        cur_val_data = (
            None if val_df is None else val_df.loc[:, cur_keys].values.ravel()
        )

        # Filter out data points that are not within the specified limits
        # Added to prevent outliers extending x-axis to far
        if xlim_n_std is not None:
            min_x, max_x = __get_min_max_x(cur_train_data, xlim_n_std)
            train_mask = (cur_train_data > min_x) & (cur_train_data < max_x)

            sns.distplot(cur_train_data[train_mask], kde=False, ax=ax)
            if val_df is not None:
                val_mask = (cur_val_data > min_x) & (cur_val_data < max_x)
                sns.distplot(cur_val_data[val_mask], kde=False, ax=ax)
        else:
            sns.distplot(cur_train_data, kde=False, ax=ax)
            if val_df is not None:
                sns.distplot(cur_val_data, kde=False, ax=ax)

        ax.set_xlabel(cur_label)

    fig.suptitle("Residuals across all IMs")

    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close()


def _get_ims(eval_result: evaluation.EvaluationResult):
    """Get the different IMs predicted"""
    return np.unique(
        [
            col.split("_")[0]
            if not col.startswith("pSA")
            else "_".join(col.split("_")[0:2])
            for col in eval_result.ln_res_train.columns
        ]
    )


def get_station_from_id(ids: np.ndarray) -> List[str]:
    """Computes the stations from station_rupture ids"""
    return [cur_split[0] for cur_split in np.char.split(ids, "_")]


def get_station_lookup(X: pd.DataFrame):
    """Creates a station - id lookup dataframe"""
    X = X.loc[:, ["lon", "lat"]].copy()

    X["station"] = get_station_from_id(X.index.values.astype(str))
    X.drop_duplicates("station", inplace=True)
    station_lookup = X.set_index("station")

    return station_lookup


def gen_spatial_data_csv(
    output_dir: str,
    eval_result: evaluation.EvaluationResult,
    ims: Iterable[str],
    agg_func: Callable = np.mean,
):
    """Generates the spatial data for GMT plotting

    For the combined residual, the individual IM residuals
    are summed.

    Parameters
    ----------
    output_dir: str
    eval_result: EvaluationResult
    ims: iterable of strings
        IMs of interest
    agg_func: callable
        The aggregation function to use to
        aggregate the data at each location

    Returns
    ----------
    combined_csv_files: list of strings
        File paths of the csv file for the combined
        IM residuals
    im_csv_files: list of strings
        File paths of the csv files for the individual
        IM type residuals
    """
    station_lookup = get_station_lookup(eval_result.training_result.X)

    combined_csv_files, im_csv_files = [], []
    for cur_df, prefix in zip(
        [eval_result.ln_res_train, eval_result.ln_res_val], ["train", "val"]
    ):
        cur_df = cur_df.copy()

        # Add location data
        cur_df["station"] = get_station_from_id(cur_df.index.values.astype(str))

        # Sum across all IMs
        cur_df["ln_res_mu"] = np.sum(
            cur_df.loc[:, [f"{im}_mean" for im in ims]], axis=1
        )
        cur_df["ln_res_sigma"] = np.sum(
            cur_df.loc[:, [f"{im}_std" for im in ims]], axis=1
        )

        cur_loc_res = cur_df.groupby("station").agg(agg_func)
        cur_loc_res = pd.merge(
            cur_loc_res, station_lookup, how="left", left_index=True, right_index=True
        )

        cur_loc_res.to_csv(os.path.join(output_dir, "loc_ln_res.csv"))

        # Create the csv & option files for summed up residuals across all IMs
        cur_output_ffp = os.path.join(output_dir, f"{prefix}_loc_ln_res_mu")
        combined_csv_files.append(
            _gmt_save(cur_loc_res, "ln_res_mu", cur_output_ffp, label="ln_res_mu")
        )

        cur_output_ffp = os.path.join(output_dir, f"{prefix}_loc_ln_res_sigma")
        combined_csv_files.append(
            _gmt_save(cur_loc_res, "ln_res_sigma", cur_output_ffp, label="ln_res_sigma")
        )

        # Create the csv & option files for each IM type
        for im in ims:
            im_csv_files.append(
                _gmt_save(
                    cur_loc_res,
                    f"{im}_mean",
                    os.path.join(
                        output_dir, f"{prefix}_loc_{im.replace('.', 'p')}_ln_res_mu"
                    ),
                    f"{im}_ln_res_mu",
                )
            )
            im_csv_files.append(
                _gmt_save(
                    cur_loc_res,
                    f"{im}_std",
                    os.path.join(
                        output_dir, f"{prefix}_loc_{im.replace('.', 'p')}_ln_res_sigma"
                    ),
                    f"{im}_ln_res_sigma",
                )
            )

    return combined_csv_files, im_csv_files


def _gmt_save(df: pd.DataFrame, key: str, output_ffp: str, label: str = None):
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
    label: str, optional
        Plot label

    Returns
    ----------
    string
        Name of the ouput csv file
    """
    df.loc[:, ["lon", "lat", key]].rename(columns={key: "value"}).to_csv(
        f"{output_ffp}.csv"
    )

    options = {} if label is None else {"title": label, "xyz-cpt-labels": label}
    with open(f"{output_ffp}.yaml", "w") as f:
        yaml.safe_dump(gen_gmt_options_dict(options=options, data_series=df[key]), f)

    return f"{output_ffp}.csv"


def gen_gmt_options_dict(
    flags: List[str] = None,
    options: Dict[str, Any] = None,
    data_series: pd.Series = None,
):
    """Generates the GMT plot options dict"""
    cur_dict = TEMPLATE_OPTIONS_DICT.copy()

    if flags is not None:
        cur_dict["flags"] = flags

    if options is not None:
        cur_dict["options"] = options

    if data_series is not None:
        std = data_series.std()

        cpt_max = float(np.round(3 * std, 1))

        options["xyz-cpt-max"], options["xyz-cpt-min"] = cpt_max, -cpt_max

        # Round down to 1 decimal places, so that there are 4 ticks on each
        # side of the colorbar
        tick_inc = float(np.round((cpt_max / 4) - 0.05, 1))
        options["xyz-cpt-tick"], options["xyz-cpt-inc"] = tick_inc, tick_inc / 2

    return cur_dict


def visualisation(
    eval_result: evaluation.EvaluationResult,
    hist_x_lim: float = None,
    vis_output_dir: str = None,
    plot_items_ffp: str = None,
    gen_options_ffp: str = None,
    plot_spatial_comb_res: bool = False,
    plot_spatial_im_res: bool = False,
    n_procs: int = 4,
):
    """"""
    print(
        f"=============================== Visualisation ==============================="
    )

    if vis_output_dir is None:
        output_dir = os.path.join(
            eval_result.training_result.output_dir, "visualisation"
        )
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    else:
        output_dir = vis_output_dir

    # # Get the different IMs predicted
    ims = _get_ims(eval_result)

    # Create a residual histogram for each IM
    print("Creating residual histograms for each IM")
    for im in ims:
        create_IM_res_hist(
            os.path.join(output_dir, f"ln_res_{im}.png"),
            eval_result.ln_res_train,
            im,
            eval_result.ln_res_val,
            xlim_n_std=hist_x_lim,
        )

    print("Creating residual histogram across all IMs")
    create_res_hist(
        os.path.join(output_dir, "ln_res.png"),
        eval_result.ln_res_train,
        ims,
        val_df=eval_result.ln_res_val,
        xlim_n_std=hist_x_lim,
    )

    # Create csv for spatial plotting
    print("Generating spatial residual plotting data")
    comb_csv_files, im_csv_files = gen_spatial_data_csv(output_dir, eval_result, ims)

    if (
        plot_items_ffp is None
        or gen_options_ffp is None
        and (plot_spatial_comb_res or plot_spatial_im_res)
    ):
        print(
            "Require path to the plot_items.py script in order to run "
            "spatial residual plotting."
        )
        return

    with open(gen_options_ffp, "r") as f:
        gen_options_dict = yaml.safe_load(f)

    if plot_spatial_comb_res:
        print("Plotting combined IM spatial residual plots")
        plot_multiple(plot_items_ffp, gen_options_dict, in_ffps=comb_csv_files,
                      n_procs=n_procs, no_clobber=True)

    if plot_spatial_im_res:
        print("Plotting IM spatial residual plots")
        plot_multiple(plot_items_ffp, gen_options_dict, in_ffps=im_csv_files,
                      n_procs=n_procs, no_clobber=True)
