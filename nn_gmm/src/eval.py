import multiprocessing as mp
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf

import ml_tools
from . import data
from .model import GMM
from .ResultDB import ResultDB
from .console import console
from . import utils

MAGNITUDE_BINS = np.arange(3, 10)
RRUP_BINS = np.arange(0, 500, 50, dtype=int)

DEFAULT_METRICS = ("bias", "sigma", "mae", "mse")
ALL_SPATIAL_METRICS = (
    "bias",
    "sigma",
    "mean_abs_residual",
    "count",
    "sum_squared_residual",
)

FANCY_METRICS = dict(
    bias=r"Bias, $\frac{1}{N} [\Sigma^N (lnIM - ln\hat{IM})]$",
    sigma=r"Error Residual, $\sigma_{\Delta_{i, j}}$",
    mae="MAE",
    mse="MSE",
)

FANCY_SPATIAL_METRICS = dict(bias=r"Bias, $\frac{1}{N} [\Sigma^N (lnIM - ln\hat{IM})]$",
                             sigma=r"$\sigma_{\Delta}$",
                             mean_abs_residual="MAE",
                             count="Count, N",
                             sum_squared_residual="Sum of Squared Error")

DEFAULT_EVAL_SITES = dict(
    chch_site="02007fb",
    nelson_site="02008b5",
    wellington_site="0200ab4",
    blenheim_site="020099b",
)

DEFAULT_CONST_FEATURES = dict(
    mag=7.0,
    dip=90,
    rake=0,
    vs30=450,
    ztor=0,
    vs500=1.5,
    z1p0=0.05,
    z2p5=0.25,
    theta=45,
    s=30,
    tect_type="ACTIVE_SHALLOW",
    rrup=50,
    rjb=50,
    rx=50,
    ry=50,
    # CCCC (I think)
    lat=-43.53145848236242,
    lon=172.63054396033107,
    X=-0.8890043834110928,
    Y=0.14077926825788484,
    Z=0.4357205571288063,
)


def write_predictions(
    model_dir: Path, data_dir: Path, output_ffp: Path, batch_size: int = 1_000_000
):
    """
    Runs predictions for the specified directory and model, and saves
    as a ResultDB

    Parameters
    ----------
    model_dir: Path
    data_dir: Path
        Directory that contains the tfrecord files to run prediction on
    output_ffp: Path
    """
    gmm = GMM.load(model_dir)

    console.print("Running predictions")
    sim_df, est_df, _ = gmm.predict_dirs(
        [data_dir],
        features=list(gmm.features),
        metadata=["lat", "lon"],
        batch_size=batch_size,
    )
    assert np.all(sim_df.index == est_df.index)

    for cur_col, cur_data in est_df.iteritems():
        sim_df[f"{cur_col}_est"] = cur_data
    del est_df

    console.print("Computing station names")
    sim_df["site"] = sim_df.index.str.rsplit("_", n=1, expand=True).get_level_values(1)

    console.print("Computing fault names")
    sim_df["fault"] = sim_df.index.str.split("_", n=1, expand=True).get_level_values(0)

    console.print("Computing rupture names")
    sim_df["rupture"] = sim_df.index.str.rsplit("_", n=1, expand=True).get_level_values(
        0
    )

    console.print("Writing database")
    ResultDB.write_data(sim_df, output_ffp)


def write_train_val_predictions(
    data_dir: Path, model_dir: Path, verbose: bool = True, batch_size: int = 1_000_000
):
    """Writes the estimated value (along with "true" values and features)
     for the specified training data directory

     Note: The data directory is expected to have
     two subdirectories "train" and "val" which each contain
     their respective tfrecord files
     """
    train_data_dir = data_dir / "train"
    val_data_dir = data_dir / "val"
    if not train_data_dir.exists() or not val_data_dir.exists():
        console.print("[red]Training or Validation directory missing[/]")
        return

    if verbose:
        console.print("Running training data predictions")
    write_predictions(
        model_dir,
        train_data_dir,
        model_dir / "train_predictions.hdf5",
        batch_size=batch_size,
    )

    if verbose:
        console.print("Running validation data predictions")
    write_predictions(
        model_dir,
        val_data_dir,
        model_dir / "val_predictions.hdf5",
        batch_size=batch_size,
    )


def mae(y: np.ndarray, y_est: np.ndarray):
    """Compute Mean Absolute Error"""
    return np.mean(np.abs(y - y_est))

def mse(y: np.ndarray, y_est: np.ndarray):
    """Computes MSE"""
    return np.mean(np.power(y - y_est, 2))


@tf.function
def tf_mse(y: tf.Tensor, y_est: tf.Tensor):
    return tf.square(y - y_est)


def mean_absolute_ln_ratio(y: np.ndarray, y_est: np.ndarray):
    """Computes the mean absolute log ratio

    Note: Assumes y & y_est are already in log-space
    """
    return np.mean(np.abs(y - y_est))


def sigma_ln_ratio(y: np.ndarray, y_est: np.ndarray):
    """Computes the standard deviation of the residual (in log-space)

    Note: Assumes both y & y_est are already in log-space
    """
    return np.std(y - y_est)


def bias(y: np.ndarray, y_est: np.ndarray):
    """Computes the average bias, i.e. mean of residual"""
    return np.mean(y - y_est)


def compute_spatial_metrics(
    data_df: pd.DataFrame,
    ims: Sequence[str],
    metrics: Sequence[str] = ALL_SPATIAL_METRICS,
):
    """Computes the specified metrics for each site"""
    metric_results = {}

    obs_keys, est_keys = ims, [f"{cur_im}_est" for cur_im in ims]
    residuals_df = data_df.loc[:, obs_keys] - data_df.loc[:, est_keys].values
    residuals_df["lon"] = data_df["lon"]
    residuals_df["lat"] = data_df["lat"]

    residuals_grouped = residuals_df.groupby(["lon", "lat"])
    if "bias" in metrics:
        metric_results["bias"] = residuals_grouped.mean()
    if "sum_squared_residual" in metrics:
        squared_residuals_df = residuals_df.copy()
        squared_residuals_df[obs_keys] = squared_residuals_df[obs_keys] ** 2
        squared_residuals_grouped = squared_residuals_df.groupby(["lon", "lat"])

        metric_results["sum_squared_residual"] = squared_residuals_grouped.sum()
    if "mean_abs_residual" in metrics:
        metric_results["mean_abs_residual"] = (
            pd.concat(
                (residuals_df.loc[:, ims].abs(), residuals_df.loc[:, ["lat", "lon"]]),
                axis=1,
            )
            .groupby(["lon", "lat"])
            .mean()
        )
    if "sigma" in metrics:
        metric_results["sigma"] = residuals_grouped.std()
    # Number of data points at each location
    if "count" in metrics:
        metric_results["count"] = residuals_grouped.count()

    return metric_results


def comp_basin_metrics(
    data_df: pd.DataFrame,
    basin_dict: Dict[str, np.ndarray],
    ims: Sequence[str],
    mag_bins: Sequence[float] = MAGNITUDE_BINS,
    rrup_bins: Sequence[int] = RRUP_BINS,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_procs: int = 4,
):
    """Computes basin metrics"""
    with mp.Pool(n_procs) as p:
        results = p.starmap(
            compute_metrics,
            [
                (
                    data_df.loc[utils.pandas_isin(data_df.site.values, cur_stations)],
                    ims,
                    mag_bins,
                    rrup_bins,
                    metrics,
                    data_df.shape[0],
                )
                for cur_basin, cur_stations in basin_dict.items()
            ],
        )

    metric_results = {
        cur_key: cur_result for cur_result, cur_key in zip(results, basin_dict.keys())
    }

    return metric_results


def _compute_metrics(
    metrics: Sequence[str],
    key_suffix: str,
    y: np.ndarray,
    y_est: np.ndarray,
    n_samples: int,
    add_count: bool,
) -> Tuple[Dict, Dict]:
    cur_metrics, counts = {}, {}

    # Standard MSE
    if "mse" in metrics:
        cur_metrics[f"mse{key_suffix}"] = mse(y, y_est)

        if add_count:
            counts[f"mse{key_suffix}"] = y.shape[0] / n_samples

    # Mean Absolute error
    if "mae" in metrics:
        cur_metrics[f"mae{key_suffix}"] = mae(y, y_est)

        if add_count:
            counts[f"mae{key_suffix}"] = y.shape[0] / n_samples

    # Mean absolute log ratio
    if "malr" in metrics:
        cur_metrics[f"malr{key_suffix}"] = mean_absolute_ln_ratio(y, y_est)

        if add_count:
            counts[f"malr{key_suffix}"] = y.shape[0] / n_samples

    # Standard deviation of residual (in log-space)
    if "sigma" in metrics:
        cur_metrics[f"sigma{key_suffix}"] = sigma_ln_ratio(y, y_est)

        if add_count:
            counts[f"sigma{key_suffix}"] = y.shape[0] / n_samples

    # Bias of predictions (i.e. mean of residuals)
    if "bias" in metrics:
        cur_metrics[f"bias{key_suffix}"] = bias(y, y_est)

        if add_count:
            counts[f"bias{key_suffix}"] = y.shape[0] / n_samples

    return cur_metrics, counts


def compute_metrics(
    data_df: pd.DataFrame,
    ims: Sequence[str],
    mag_bins: Sequence[float] = MAGNITUDE_BINS,
    rrup_bins: Sequence[int] = RRUP_BINS,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_samples: int = None,
):
    """
    Computes a bunch of metrics for the specified IMs

    Parameters
    ----------
    data_df: Dataframe
    ims: list of strings
    rrup_bins: Sequence of floats, optional
        Magnitude bins for which to compute the metrics
    rrup_bins: Sequence of ints, optional
        Magnitude bins for which to compute the metrics
    metrics: Sequence of strings, optional
        Metrics to compute
    n_samples: int, optional
        Number of samples to use for computing relative count
    Returns
    -------
    Dictionary of evaluation metrics
    """
    # Compute the statistics

    metric_results = {}
    n_samples = data_df.shape[0] if n_samples is None else n_samples
    for ix, cur_im in enumerate(ims):
        y, y_est = data_df[cur_im], data_df[f"{cur_im}_est"]

        cur_metrics, counts = _compute_metrics(
            metrics, "", y, y_est, n_samples, ix == 0
        )

        # Compute metric per magnitude bin
        mag_bin_ind = np.digitize(data_df.mag, bins=mag_bins)
        for cur_bin_ind in np.unique(mag_bin_ind):
            cur_mask = mag_bin_ind == cur_bin_ind

            cur_result = _compute_metrics(
                metrics,
                f"_mag_{mag_bins[cur_bin_ind - 1]}_{mag_bins[cur_bin_ind]}",
                y[cur_mask],
                y_est[cur_mask],
                n_samples,
                ix == 0,
            )
            cur_metrics = cur_metrics | cur_result[0]
            counts = counts | cur_result[1]

        # Compute metric per rrup bin
        rrup_bin_ind = np.digitize(data_df.rrup, bins=rrup_bins)
        for cur_bin_ind in np.unique(rrup_bin_ind):
            cur_mask = rrup_bin_ind == cur_bin_ind

            cur_result = _compute_metrics(
                metrics,
                f"_rrup_{rrup_bins[cur_bin_ind - 1]}_{rrup_bins[cur_bin_ind]}",
                y[cur_mask],
                y_est[cur_mask],
                n_samples,
                ix == 0,
            )
            cur_metrics = cur_metrics | cur_result[0]
            counts = counts | cur_result[1]

        metric_results["count"] = counts
        metric_results[cur_im] = cur_metrics

    return pd.DataFrame.from_dict(metric_results)


def comp_train_val_metrics(
    model_dir: Path, save: bool = False, metrics: Sequence[str] = DEFAULT_METRICS
):
    """Computes training & validation metrics"""
    ims = GMM.load(model_dir).ims
    columns = [f"{im}_est" for im in ims] + ims + ["mag", "rrup"]

    train_df = ResultDB.get_data_static(model_dir / "train_predictions.hdf5", columns)
    val_df = ResultDB.get_data_static(model_dir / "val_predictions.hdf5", columns)

    train_metrics = compute_metrics(train_df, ims, metrics=metrics)
    val_metrics = compute_metrics(val_df, ims, metrics=metrics)

    console.rule("Training")
    console.print(train_metrics)

    console.rule("Validation")
    console.print(val_metrics)

    if save:
        train_metrics.to_csv(model_dir / "train_metrics.csv")
        val_metrics.to_csv(model_dir / "val_metrics.csv")

    return train_metrics, val_metrics


def comp_train_val_spatial_metrics(
    model_dir: Path, save: bool = False, metrics: Sequence[str] = ALL_SPATIAL_METRICS
):
    """Computes the specified metrics for each site,
    for both training and validation data"""
    ims = GMM.load(model_dir).ims

    columns = [f"{im}_est" for im in ims] + list(ims) + ["site", "lat", "lon"]
    train_df = ResultDB.get_data_static(model_dir / "train_predictions.hdf5", columns)
    val_df = ResultDB.get_data_static(model_dir / "val_predictions.hdf5", columns)

    train_spatial_metrics = compute_spatial_metrics(train_df, ims, metrics=metrics)
    val_spatial_metrics = compute_spatial_metrics(val_df, ims, metrics=metrics)

    if save:
        train_out_dir = model_dir / "train_spatial_metrics"
        val_out_dir = model_dir / "val_spatial_metrics"
        train_out_dir.mkdir(exist_ok=True)
        val_out_dir.mkdir(exist_ok=True)

        for cur_metric in train_spatial_metrics.keys():
            train_spatial_metrics[cur_metric].to_csv(
                train_out_dir / f"{cur_metric}.csv"
            )
            val_spatial_metrics[cur_metric].to_csv(val_out_dir / f"{cur_metric}.csv")

    return train_spatial_metrics, val_spatial_metrics


def comp_train_val_basin_metrics(
    model_dir: Path,
    basin_dir: Path,
    save: bool = False,
    print_metrics: bool = True,
    mag_bins: Sequence[float] = MAGNITUDE_BINS,
    rrup_bins: Sequence[int] = RRUP_BINS,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Computes basin metrics for the training & validation data"""
    # Get the model IMs
    ims = list(
        ml_tools.utils.load_json(model_dir / "input_config.json")["im_config"].keys()
    )
    columns = ims + [f"{im}_est" for im in ims] + ["site", "mag", "rrup"]

    # Get the basin stations
    basin_dict = data.load_basin_stations(basin_dir)

    # Retrieve model predictions
    console.print("Loading training model predictions")
    train_data_df = ResultDB.get_data_static(
        model_dir / "train_predictions.hdf5", columns
    )

    console.print("Loading validation model predictions")
    val_data_df = ResultDB.get_data_static(model_dir / "val_predictions.hdf5", columns)

    # Compute the basin metrics
    console.print("Compute metrics")
    train_metrics = comp_basin_metrics(
        train_data_df, basin_dict, ims, mag_bins=mag_bins, rrup_bins=rrup_bins, metrics=metrics
    )
    val_metrics = comp_basin_metrics(
        val_data_df, basin_dict, ims, mag_bins=mag_bins, rrup_bins=rrup_bins, metrics=metrics
    )

    # Add metrics for base-grid stations
    train_metrics["BaseGrid"] = compute_metrics(
        train_data_df.loc[
            np.char.startswith(train_data_df.site.values.astype(str), "0")
        ],
        ims,
        n_samples=train_data_df.shape[0],
        mag_bins=mag_bins,
        rrup_bins=rrup_bins,
        metrics=metrics,
    )
    val_metrics["BaseGrid"] = compute_metrics(
        val_data_df.loc[np.char.startswith(val_data_df.site.values.astype(str), "0")],
        ims,
        n_samples=val_data_df.shape[0],
        mag_bins=mag_bins,
        rrup_bins=rrup_bins,
        metrics=metrics,
    )

    # Save & print
    if save:
        (model_dir / "train_basin_metrics").mkdir(exist_ok=True)
        (model_dir / "val_basin_metrics").mkdir(exist_ok=True)
    for cur_basin in train_metrics.keys():
        if print_metrics:
            console.rule(cur_basin)
            console.print(
                pd.merge(
                    train_metrics[cur_basin],
                    val_metrics[cur_basin],
                    left_index=True,
                    right_index=True,
                    suffixes=("_train", "_val"),
                )
            )
        if save:
            train_metrics[cur_basin].to_csv(
                model_dir / "train_basin_metrics" / f"{cur_basin}.csv"
            )
            val_metrics[cur_basin].to_csv(
                model_dir / "val_basin_metrics" / f"{cur_basin}.csv"
            )

    return train_metrics, val_metrics


def wandb_log_metrics(
    wandb_run: object,
    ims: Sequence[str],
    prefix: str,
    general_metrics: pd.DataFrame,
    basin_metrics: Dict[str, pd.DataFrame],
):
    """Logs the given metrics to wandb"""
    for cur_im in ims:
        # cur_im_name = cur_im.replace(".", "p")

        # Write general metrics
        cur_metrics = general_metrics[cur_im]
        cur_metrics.index = np.char.add(
            f"{prefix}_{cur_im}_", cur_metrics.index.values.astype(str)
        )
        wandb_run.summary.update(cur_metrics.to_dict())

        # Write basin metrics
        for cur_basin in basin_metrics.keys():
            cur_metrics = basin_metrics[cur_basin][cur_im]
            cur_metrics.index = np.char.add(
                f"{prefix}_{cur_im}_{cur_basin}_", cur_metrics.index.values.astype(str)
            )
            wandb_run.summary.update(cur_metrics.to_dict())


def load_basin_metrics(
    model_dir: Path, train_only: bool = False, val_only: bool = False
):
    """Loads already computed basin metrics
    for both training and validation data"""
    assert (train_only and not val_only) or (val_only and not train_only)

    train_basin_metrics = {
        cur_ffp.stem: pd.read_csv(cur_ffp, index_col=0)
        for cur_ffp in (model_dir / "train_basin_metrics").glob("*.csv")
    }
    val_basin_metrics = {
        cur_ffp.stem: pd.read_csv(cur_ffp, index_col=0)
        for cur_ffp in (model_dir / "val_basin_metrics").glob("*.csv")
    }

    if train_only:
        return train_basin_metrics
    elif val_only:
        return val_basin_metrics
    else:
        return train_basin_metrics, val_basin_metrics


def load_spatial_metrics(model_dir: Path):
    """Loads alread computed spatial metrics
    for both training and validation data"""
    train_spatial_metrics = {
        cur_ffp.stem: pd.read_csv(cur_ffp)
        for cur_ffp in (model_dir / "train_spatial_metrics").glob("*.csv")
    }
    val_spatial_metrics = {
        cur_ffp.stem: pd.read_csv(cur_ffp)
        for cur_ffp in (model_dir / "val_spatial_metrics").glob("*.csv")
    }

    # Drop NaN-values for Sigma
    if "sigma" in train_spatial_metrics.keys():
        console.print("Dropping NaN-values for spatial sigma")
        train_spatial_metrics["sigma"].dropna(inplace=True)
        val_spatial_metrics["sigma"].dropna(inplace=True)

        console.print(
            f"\tGiving: \n"
            f"\t\tTraining - {train_spatial_metrics['sigma'].shape[0]} stations\n"
            f"\t\tValidation - {val_spatial_metrics['sigma'].shape[0]} stations"
        )

    return train_spatial_metrics, val_spatial_metrics
