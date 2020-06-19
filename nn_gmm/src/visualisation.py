import os
from typing import Tuple, Iterable, Callable, Dict, List, Any

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from visualization.gmt.plotting import plot_multiple
from .evaluation import EvaluationResult
from .training import TrainingResult

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


class PlotGen:

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
            "xyz-cpt-inc": "0.25",
            "xyz-cpt-tick": "0.5",
            "xyz-cpt-min": "-2.0",
            "xyz-cpt-max": "2.0",
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

    LN_RES_IM_FNAME_TEMPLATE = "ln_res_{}.png"

    def __init__(self, eval_result: EvaluationResult, output_dir: str = None):
        """Constructor for Visualisation"""
        self.eval_result = eval_result
        self.train_result = eval_result.training_result
        self.output_dir = (
            output_dir
            if output_dir is not None
            else os.path.join(eval_result.training_result.output_dir, "visualisation")
        )

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.train_res_df = eval_result.ln_res_train
        self.val_res_df = eval_result.ln_res_val

        self.ims = self._get_IMs()

        # Spatial csv files for GMT plotting
        # Don't use these variables directly, use the properties instead (lazy loading)
        self._comb_res_csv_files, self._im_res_csv_files = None, None
        self._im_sigma_csv_files = None
        self._n_ruptures_csv_files = None

    @property
    def comb_res_csv_files(self):
        if self._comb_res_csv_files is None:
            print(f"Computing spatial residual data")
            self._comb_res_csv_files, self._im_res_csv_files = gen_spatial_data_res_csv(
                self.output_dir, self.eval_result, self.ims
            )
        return self._comb_res_csv_files

    @property
    def im_res_csv_files(self):
        if self._im_res_csv_files is None:
            print(f"Computing spatial residual data")
            self._comb_res_csv_files, self._im_res_csv_files = gen_spatial_data_res_csv(
                self.output_dir, self.eval_result, self.ims
            )
        return self._im_res_csv_files

    @property
    def im_sigma_csv_files(self):
        if self._im_sigma_csv_files is None:
            print(f"Computing spatial sigma data")
            self._im_sigma_csv_files = gen_spatial_sigma_csv(
                self.output_dir, self.eval_result, self.ims
            )

        return self._im_sigma_csv_files

    @property
    def n_ruptures_csv_files(self):
        if self._n_ruptures_csv_files is None:
            print(f"Computing spatial n_ruptures data")
            self._n_ruptures_csv_files = gen_spatial_nruptures_csv(
                self.output_dir, self.eval_result
            )

        return self._n_ruptures_csv_files

    def _get_IMs(self):
        """Get the different IMs predicted"""
        return np.unique(
            [
                col.split("_")[0]
                if not col.startswith("pSA")
                else "_".join(col.split("_")[0:2])
                for col in self.train_res_df.columns
            ]
        )

    def create_IM_res_hist(
        self, im: str, output_ffp: str = None, xlim_n_std: float = None
    ):
        """Creates a ln residual for the specified IM"""
        output_ffp = (
            output_ffp
            if output_ffp is not None
            else os.path.join(self.output_dir, self.LN_RES_IM_FNAME_TEMPLATE.format(im))
        )
        return create_IM_res_hist(
            output_ffp, self.train_res_df, im, self.val_res_df, xlim_n_std=xlim_n_std
        )

    def create_IM_res_hists(
        self,
        ims: Iterable[str] = None,
        output_dir: str = None,
        xlim_n_std: float = None,
    ):
        """Creates a residual histogram for each IM type"""
        ims = ims if ims is not None else self.ims
        output_dir = output_dir if output_dir is not None else self.output_dir

        print("Creating ln residual histograms for each IM")
        for im in ims:
            self.create_IM_res_hist(
                im,
                os.path.join(output_dir, self.LN_RES_IM_FNAME_TEMPLATE.format(im)),
                xlim_n_std=xlim_n_std,
            )

    def create_comb_res_hist(
        self,
        ims: Iterable[str] = None,
        output_ffp: str = None,
        xlim_n_std: float = None,
    ):
        """Creates a residual histogram across all the specified
        IM types (defaults to all)"""
        output_ffp = (
            output_ffp
            if output_ffp is not None
            else os.path.join(self.output_dir, "ln_res.png")
        )
        ims = ims if ims is not None else self.ims

        return create_res_hist(
            output_ffp,
            self.train_res_df,
            ims,
            val_df=self.val_res_df,
            xlim_n_std=xlim_n_std,
        )

    def create_comb_res_maps(
        self,
        plot_items_ffp: str,
        gen_options_dict: Dict = None,
        n_procs: int = 4,
        no_clobber: bool = True,
    ):
        """Creates the residual maps for the combined IMs (mean)
        Requires GMT to be setup
        """
        gen_options_dict = (
            gen_options_dict
            if gen_options_dict is not None
            else self.DEFAULT_RES_GEN_GMT_PLOT_OPTIONS
        )

        return plot_multiple(
            plot_items_ffp,
            gen_options_dict,
            in_ffps=self.comb_res_csv_files,
            n_procs=n_procs,
            no_clobber=no_clobber,
        )

    def create_IM_res_maps(
        self,
        plot_items_ffp: str,
        gen_options_dict: Dict = None,
        n_procs: int = 4,
        no_clobber: bool = True,
    ):
        """Creates a residual map for each IM
        Requires GMT to be setup"""
        gen_options_dict = (
            gen_options_dict
            if gen_options_dict is not None
            else self.DEFAULT_RES_GEN_GMT_PLOT_OPTIONS
        )

        return plot_multiple(
            plot_items_ffp,
            gen_options_dict,
            in_ffps=self.im_res_csv_files,
            n_procs=n_procs,
            no_clobber=no_clobber,
        )

    def create_sigma_maps(
        self,
        plot_items_ffp: str,
        gen_options_dict: Dict = None,
        n_procs: int = 4,
        no_clobber: bool = True,
    ):
        gen_options_dict = (
            gen_options_dict
            if gen_options_dict is not None
            else self.DEFAULT_STANDARD_GMT_PLOT_OPTIONS
        )

        plot_multiple(
            plot_items_ffp,
            gen_options_dict,
            in_ffps=self.im_sigma_csv_files,
            n_procs=n_procs,
            no_clobber=no_clobber,
        )

    def create_nruptures_maps(
        self,
        plot_items_ffp: str,
        gen_options_dict: Dict = None,
        n_procs: int = 4,
        no_clobber: bool = True,
    ):
        gen_options_dict = (
            gen_options_dict
            if gen_options_dict is not None
            else self.DEFAULT_STANDARD_GMT_PLOT_OPTIONS
        )

        plot_multiple(
            plot_items_ffp,
            gen_options_dict,
            in_ffps=self.n_ruptures_csv_files,
            n_procs=n_procs,
            no_clobber=no_clobber,
        )


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


def gen_spatial_nruptures_csv(output_dir: str, eval_result: EvaluationResult):
    station_lookup = get_station_lookup(eval_result.training_result.X)

    csv_files = []
    for cur_df, prefix in zip(
        [eval_result.y_train_est, eval_result.y_val_est, eval_result.training_result.X],
        ["train", "val", "all"],
    ):
        cur_df["station"] = get_station_from_id(cur_df.index.values.astype(str))

        cur_loc_count = cur_df.groupby("station").count().iloc[:, 0]
        cur_loc_count.name = "n_ruptures"

        cur_loc_count = pd.merge(
            cur_loc_count, station_lookup, how="left", left_index=True, right_index=True
        )

        gmt_options = get_gmt_options_dict(
            options={
                **{"title": f"{prefix}_n_ruptures", "xyz-cpt-labels": f"n_ruptures"},
                **compute_GMT_std_ticks(cur_loc_count["n_ruptures"], n_std=2),
            }
        )
        csv_files.append(
            _gmt_save(
                cur_loc_count,
                "n_ruptures",
                os.path.join(output_dir, f"{prefix}_n_ruptures"),
                gmt_options=gmt_options,
            )
        )

    return csv_files


def gen_spatial_sigma_csv(
    output_dir: str,
    eval_result: EvaluationResult,
    ims: Iterable[str],
    agg_func: Callable = np.nanmean,
):
    """Generates the spatial sigma data for GMT plotting

    Parameters
    ----------
    output_dir: str
    eval_result: EvaluationResult
    ims: iterable of strings
        IMs of interest
    agg_func: callable
        The aggregation function to use to
        aggregate the data at each location
        Make sure this function can handle nan
        values..
    """
    station_lookup = get_station_lookup(eval_result.training_result.X)

    csv_files = []
    for cur_df, prefix in zip(
        [eval_result.y_train_est, eval_result.y_val_est], ["train", "val"]
    ):
        cur_df["station"] = get_station_from_id(cur_df.index.values.astype(str))

        cur_loc_sigma = cur_df.groupby("station").agg(agg_func)

        cur_loc_sigma = pd.merge(
            cur_loc_sigma, station_lookup, how="left", left_index=True, right_index=True
        )

        for im in ims:
            gmt_options = get_gmt_options_dict(
                options={
                    **{"title": f"{prefix}_{im}_std", "xyz-cpt-labels": f"{im}_std"},
                    **compute_GMT_std_ticks(cur_loc_sigma[f"{im}_std"], n_std=3),
                }
            )
            csv_files.append(
                _gmt_save(
                    cur_loc_sigma,
                    f"{im}_std",
                    os.path.join(output_dir, f"{prefix}_{im.replace('.', 'p')}_std"),
                    gmt_options=gmt_options,
                )
            )

        return csv_files


def gen_spatial_data_res_csv(
    output_dir: str,
    eval_result: EvaluationResult,
    ims: Iterable[str],
    agg_func: Callable = np.nanmean,
):
    """Generates the spatial residual data for GMT plotting

    For the combined residual, the individual IM residuals
    are averaged (mean).

    Parameters
    ----------
    output_dir: str
    eval_result: EvaluationResult
    ims: iterable of strings
        IMs of interest
    agg_func: callable
        The aggregation function to use to
        aggregate the data at each location
        Make sure this function can handle nan
        values..

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
        cur_df["ln_res_mu"] = np.nanmean(
            cur_df.loc[:, [f"{im}_mean" for im in ims]], axis=1
        )
        cur_df["ln_res_sigma"] = np.nanmean(
            cur_df.loc[:, [f"{im}_std" for im in ims]], axis=1
        )

        cur_loc_res = cur_df.groupby("station").agg(agg_func)
        cur_loc_res = pd.merge(
            cur_loc_res, station_lookup, how="left", left_index=True, right_index=True
        )

        # Deal with any nan values,
        # these arise from the model predicting negative standard deviations
        cur_loc_res.values[cur_loc_res.isna()] = 100

        cur_loc_res.to_csv(os.path.join(output_dir, "loc_ln_res.csv"))

        # Create the csv & option files for summed up residuals across all IMs
        cur_output_ffp = os.path.join(output_dir, f"{prefix}_loc_ln_res_mu")
        combined_csv_files.append(
            _gmt_save(
                cur_loc_res,
                "ln_res_mu",
                cur_output_ffp,
                gmt_options=get_gmt_options_dict(
                    options={"title": f"{prefix}_ln_res_mu", "xyz-cpt-labels": "ln_res_mu"}
                ),
            )
        )

        cur_output_ffp = os.path.join(output_dir, f"{prefix}_loc_ln_res_sigma")
        combined_csv_files.append(
            _gmt_save(
                cur_loc_res,
                "ln_res_sigma",
                cur_output_ffp,
                gmt_options=get_gmt_options_dict(
                    options={"title": f"{prefix}_ln_res_sigma", "xyz-cpt-labels": "ln_res_sigma"}
                ),
            )
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
                    get_gmt_options_dict(
                        options={
                            "title": f"{prefix}_{im}_ln_res_mu",
                            "xyz-cpt-labels": f"{im}_ln_res_mu",
                        }
                    ),
                )
            )
            im_csv_files.append(
                _gmt_save(
                    cur_loc_res,
                    f"{im}_std",
                    os.path.join(
                        output_dir, f"{prefix}_loc_{im.replace('.', 'p')}_ln_res_sigma"
                    ),
                    get_gmt_options_dict(
                        options={
                            "title": f"{prefix}_{im}_ln_res_sigma",
                            "xyz-cpt-labels": f"{im}_ln_res_sigma",
                        }
                    ),
                )
            )

    return combined_csv_files, im_csv_files


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
        Name of the ouput csv file
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
    data_series: pd.Series, n_std: float = 2, center: float = None
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

    tick_inc = float(round(((cpt_max - center) / 4), n_dec_points))
    cpt_max = round(center + (tick_inc * 4), n_dec_points)

    options["xyz-cpt-max"] = cpt_max
    options["xyz-cpt-min"] = round(center - (4 * tick_inc), n_dec_points)
    options["xyz-cpt-tick"], options["xyz-cpt-inc"] = tick_inc, tick_inc / 2

    return options
