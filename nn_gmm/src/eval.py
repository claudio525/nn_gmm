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

DEFAULT_METRICS = ("bias", "sigma")


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
    lat=-43.53145848236242,
    lon=172.63054396033107,
)


def write_predictions(model_dir: Path, data_dir: Path, output_ffp: Path):
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
    sim_df, est_df, _ = gmm.predict_dirs([data_dir], features=list(gmm.features))
    assert np.all(sim_df.index == est_df.index)

    for cur_col, cur_data in est_df.iteritems():
        sim_df[f"{cur_col}_est"] = cur_data
    del est_df

    console.print("Computing station names")
    sim_df["site"] = sim_df.index.str.rsplit("_", n=1, expand=True).get_level_values(1)

    console.print("Computing event names")
    sim_df["event"] = sim_df.index.str.split("_REL", n=1, expand=True).get_level_values(
        0
    )

    console.print("Writing database")
    ResultDB.write_data(sim_df, output_ffp)


def write_train_val_predictions(data_dir: Path, model_dir: Path, verbose: bool = True):
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
    write_predictions(model_dir, train_data_dir, model_dir / "train_predictions.hdf5")

    if verbose:
        console.print("Running validation data predictions")
    write_predictions(model_dir, val_data_dir, model_dir / "val_predictions.hdf5")


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
    data_df: pd.DataFrame, ims: Sequence[str], metrics: Sequence[str] = DEFAULT_METRICS
):
    """Computes the specified metrics for each site"""
    metric_results = {}

    obs_keys, est_keys = ims, [f"{cur_im}_est" for cur_im in ims]
    residuals_df = data_df.loc[:, obs_keys] - data_df.loc[:, est_keys].values
    residuals_df["site"] = data_df["site"]

    residuals_grouped = residuals_df.groupby("site")
    if "bias" in metrics:
        metric_results["bias"] = residuals_grouped.mean()
    if "sigma" in metrics:
        metric_results["sigma"] = residuals_grouped.std()

    return metric_results


def comp_basin_metrics(
    data_df: pd.DataFrame, basin_dict: Dict[str, np.ndarray], ims: Sequence[str]
):
    """Computes basin metrics"""
    metrics = {}
    for cur_basin, cur_stations in basin_dict.items():
        cur_basin_data = data_df.loc[
            utils.pandas_isin(data_df.site.values, cur_stations)
        ]
        cur_basin_metrics = compute_metrics(
            cur_basin_data, ims, n_samples=data_df.shape[0]
        )

        metrics[cur_basin] = cur_basin_metrics

    return metrics


def compute_metrics(
    data_df: pd.DataFrame,
    ims: Sequence[str],
    mag_bins: Sequence[float] = MAGNITUDE_BINS,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_samples: int = None,
):
    """
    Computes a bunch of metrics for the specified IMs

    Parameters
    ----------
    data_df: Dataframe
    ims: list of strings
    mag_bins: Sequence of floats, optional
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
    counts = {}
    n_samples = data_df.shape[0] if n_samples is None else n_samples
    for ix, cur_im in enumerate(ims):
        cur_metrics = {}
        y, y_est = data_df[cur_im], data_df[f"{cur_im}_est"]

        # Standard MSE
        if "mse" in metrics:
            cur_metrics["mse"] = mse(y, y_est)

            if ix == 0:
                counts["mse"] = y.shape[0] / n_samples

        # Mean absolute log ratio
        if "malr" in metrics:
            cur_metrics["malr"] = mean_absolute_ln_ratio(y, y_est)

            if ix == 0:
                counts["malr"] = y.shape[0] / n_samples

        # Standard deviation of residual (in log-space)
        if "sigma" in metrics:
            cur_metrics["sigma"] = sigma_ln_ratio(y, y_est)

            if ix == 0:
                counts["sigma"] = y.shape[0] / n_samples

        # Bias of predictions (i.e. mean of residuals)
        if "bias" in metrics:
            cur_metrics["bias"] = bias(y, y_est)

            if ix == 0:
                counts["bias"] = y.shape[0] / n_samples

        # Compute metric per magnitude bin
        mag_bin_ind = np.digitize(data_df.mag, bins=mag_bins)
        for cur_bin_ind in np.unique(mag_bin_ind):
            cur_mask = mag_bin_ind == cur_bin_ind

            # MSE
            if "mse" in metrics:
                cur_key = f"mse_mag_{mag_bins[cur_bin_ind - 1]}_{mag_bins[cur_bin_ind]}"
                cur_metrics[cur_key] = mse(y[cur_mask], y_est[cur_mask])

                if ix == 0:
                    counts[cur_key] = np.count_nonzero(cur_mask) / n_samples

            # Mean absolute log ratio
            if "malr" in metrics:
                cur_key = f"malr_{mag_bins[cur_bin_ind - 1]}_{mag_bins[cur_bin_ind]}"
                cur_metrics[cur_key] = mean_absolute_ln_ratio(
                    y[cur_mask], y_est[cur_mask]
                )

                if ix == 0:
                    counts[cur_key] = np.count_nonzero(cur_mask) / n_samples

            # Standard deviation of residual (in log-space)
            if "sigma" in metrics:
                cur_key = f"sigma_{mag_bins[cur_bin_ind - 1]}_{mag_bins[cur_bin_ind]}"
                cur_metrics[cur_key] = sigma_ln_ratio(y[cur_mask], y_est[cur_mask])

                if ix == 0:
                    counts[cur_key] = np.count_nonzero(cur_mask) / n_samples

            # Bias of predictions (i.e. mean of residuals)
            if "bias" in metrics:
                cur_key = f"bias_{mag_bins[cur_bin_ind - 1]}_{mag_bins[cur_bin_ind]}"
                cur_metrics[cur_key] = bias(y[cur_mask], y_est[cur_mask])

                if ix == 0:
                    counts[cur_key] = np.count_nonzero(cur_mask) / n_samples

        metric_results["count"] = counts
        # metric_results[cur_im.replace(".", "p")] = cur_metrics
        metric_results[cur_im] = cur_metrics

    return pd.DataFrame.from_dict(metric_results)


def comp_train_val_metrics(
    model_dir: Path, save: bool = False, metrics: Sequence[str] = DEFAULT_METRICS
):
    """Computes training & validation metrics"""
    ims = GMM.load(model_dir).ims
    columns = [f"{im}_est" for im in ims] + ims + ["mag"]

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
    model_dir: Path, save: bool = False, metrics: Sequence[str] = DEFAULT_METRICS
):
    """Computes the specified metrics for each site,
    for both training and validation data"""
    ims = GMM.load(model_dir).ims

    columns = [f"{im}_est" for im in ims] + list(ims) + ["site"]
    train_df = ResultDB.get_data_static(model_dir / "train_predictions.hdf5", columns)
    val_df = ResultDB.get_data_static(model_dir / "val_predictions.hdf5", columns)

    train_spatial_metrics = compute_spatial_metrics(val_df, ims, metrics=metrics)
    val_spatial_metrics = compute_spatial_metrics(train_df, ims, metrics=metrics)

    if save:
        train_out_dir = model_dir / "train_spatial_metrics"
        val_out_dir = model_dir / "val_spatial_metrics"
        train_out_dir.mkdir(exist_ok=False)
        val_out_dir.mkdir(exist_ok=False)

        for cur_metric in train_spatial_metrics.keys():
            train_spatial_metrics[cur_metric].to_csv(
                train_out_dir / f"{cur_metric}.csv"
            )
            val_spatial_metrics[cur_metric].to_csv(val_out_dir / f"{cur_metric}.csv")

    return train_spatial_metrics, val_spatial_metrics


def comp_train_val_basin_metrics(
    model_dir: Path, basin_dir: Path, save: bool = False, print_metrics: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Computes basin metrics for the training & validation data"""
    # Get the model IMs
    ims = list(
        ml_tools.utils.load_json(model_dir / "input_config.json")["im_config"].keys()
    )
    columns = ims + [f"{im}_est" for im in ims] + ["site", "mag"]

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
    train_metrics = comp_basin_metrics(train_data_df, basin_dict, ims)
    val_metrics = comp_basin_metrics(val_data_df, basin_dict, ims)

    # Save & print
    if save:
        (model_dir / "train_basin_metrics").mkdir(parents=False, exist_ok=True)
        (model_dir / "val_basin_metrics").mkdir(parents=False, exist_ok=True)
    for cur_basin in basin_dict.keys():
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
        cur_ffp.stem: pd.read_csv(cur_ffp, index_col=0)
        for cur_ffp in (model_dir / "train_spatial_metrics").glob("*.csv")
    }
    val_spatial_metrics = {
        cur_ffp.stem: pd.read_csv(cur_ffp, index_col=0)
        for cur_ffp in (model_dir / "val_spatial_metrics").glob("*.csv")
    }

    return train_spatial_metrics, val_spatial_metrics
