from pathlib import Path
from typing import List, Union, Dict, Sequence

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ml_tools
from . import data
from .model import GMM
from .ResultDB import ResultDB
from .console import console
from . import utils
from nn_gmm.src.plotting.BinPlotGen import BinPlotGen

MAGNITUDE_BINS = np.arange(3, 10)

DEFAULT_METRICS = ("malr")


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

    console.log("Running predictions")
    sim_df, est_df, *_ = gmm.predict_dirs([data_dir], features=list(gmm.features))
    assert np.all(sim_df.index == est_df.index)

    for cur_col, cur_data in est_df.iteritems():
        sim_df[f"{cur_col}_est"] = cur_data
    del est_df

    console.log("Computing station names")
    sim_df["site"] = sim_df.index.str.rsplit("_", n=1, expand=True).get_level_values(1)
    # sim_df["site"] = utils.get_station_from_id(
    #     sim_df.index.values.astype(str)
    # )

    console.log("Writing database")
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
        console.log("[red]Training or Validation directory missing[/]")
        return

    if verbose:
        console.log("Running training data predictions")
    write_predictions(model_dir, train_data_dir, model_dir / "train_predictions.hdf5")

    if verbose:
        console.log("Running validation data predictions")
    write_predictions(model_dir, val_data_dir, model_dir / "val_predictions.hdf5")


def mse(y: np.ndarray, y_est: np.ndarray):
    """Computes MSE"""
    return np.mean(np.power(y - y_est, 2))


def mean_absolute_ln_ratio(y: np.ndarray, y_est: np.ndarray):
    """Computes the mean absolute log ratio

    Note: Assumes y & y_est are already in log-space
    """
    return np.mean(np.abs(y - y_est))


def compute_metrics(
    data_df: pd.DataFrame,
    ims: Sequence[str],
    mag_bins: Sequence[float] = MAGNITUDE_BINS,
    metrics: Sequence[str] = DEFAULT_METRICS,
):
    """
    Computes a bunch of metrics for the specified IMs

    Parameters
    ----------
    data_df: Dataframe
    ims: list of strings
    mag_bins: Sequence of floats
        Magnitude bins for which to compute MSE

    Returns
    -------
    Dictionary of evaluation metrics
    """
    # Compute the statistics
    metric_results = {}
    for cur_im in ims:
        cur_metrics = {}
        y, y_est = data_df[cur_im], data_df[f"{cur_im}_est"]

        # Standard MSE
        if "mse" in metrics:
            cur_metrics["mse"] = mse(y, y_est)

        # Mean absolute log ratio
        if "malr" in metrics:
            cur_metrics["malr"] = mean_absolute_ln_ratio(y, y_est)

        # Compute MSE per magnitude bin
        mag_bin_ind = np.digitize(data_df.mag, bins=mag_bins)
        for cur_bin_ind in np.unique(mag_bin_ind):
            cur_mask = mag_bin_ind == cur_bin_ind

            # MSE
            if "mse" in metrics:
                cur_metrics[
                    f"mse_mag_{mag_bins[cur_bin_ind - 1]}_{mag_bins[cur_bin_ind]}"
                ] = mse(y[cur_mask], y_est[cur_mask])

            # Mean absolute log ratio
            if "malr" in metrics:
                cur_metrics[
                    f"malr_{mag_bins[cur_bin_ind - 1]}_{mag_bins[cur_bin_ind]}"
                ] = mean_absolute_ln_ratio(y[cur_mask], y_est[cur_mask])

        metric_results[cur_im] = cur_metrics

    return pd.DataFrame.from_dict(metric_results)


def train_val_metrics(model_dir: Path, save: bool = False):
    """Computes training & validation metrics"""
    input_config = ml_tools.utils.load_json(model_dir / "input_config.json")

    ims = list(input_config["im_config"].keys())
    columns = [f"{im}_est" for im in ims] + ims + ["mag"]

    train_df = ResultDB.get_data_static(model_dir / "train_predictions.hdf5", columns)
    val_df = ResultDB.get_data_static(model_dir / "val_predictions.hdf5", columns)

    train_metrics = compute_metrics(train_df, ims)
    val_metrics = compute_metrics(val_df, ims)

    console.rule("Training")
    console.log(train_metrics)

    console.rule("Validation")
    console.log(val_metrics)

    if save:
        train_metrics.to_csv(model_dir / "train_metrics.csv")
        val_metrics.to_csv(model_dir / "val_metrics.csv")

    return train_metrics, val_metrics


def gen_rrup_bin_plot(model_dir: Path, im: str):
    """Creates a Rrup based plot for the specified IM"""
    output_dir = model_dir / "plots" / "bin_plots"
    val_result_db_ffp = model_dir / f"val_predictions.hdf5"
    train_result_db_ffp = model_dir / f"train_predictions.hdf5"

    bin_plot_gen = BinPlotGen()
    console.log("Generating training plots")
    bin_plot_gen.create_IM_bin_plot(
        im, output_dir, train_result_db_ffp, feature="rrup", prefix="train"
    )
    console.log("Generating validation plots")
    bin_plot_gen.create_IM_bin_plot(
        im, output_dir, val_result_db_ffp, feature="rrup", prefix="val"
    )


def compute_basin_metrics(
    data_df: pd.DataFrame, basin_dict: Dict[str, np.ndarray], ims: Sequence[str]
):
    """Computes basin metrics"""
    metrics = {}
    for cur_basin, cur_stations in basin_dict.items():
        cur_basin_data = data_df.loc[
            utils.pandas_isin(data_df.site.values, cur_stations)
        ]
        cur_basin_metrics = compute_metrics(cur_basin_data, ims)

        metrics[cur_basin] = cur_basin_metrics

    return metrics


def train_val_basin_metrics(model_dir: Path, basin_dir: Path, save: bool = False):
    """Computes basin metrics for the training & validation data"""
    # Get the model IMs
    ims = list(
        ml_tools.utils.load_json(model_dir / "input_config.json")["im_config"].keys()
    )
    columns = ims + [f"{im}_est" for im in ims] + ["site", "mag"]

    # Get the basin stations
    basin_dict = data.load_basin_stations(basin_dir)

    # Retrieve model predictions
    console.log("Loading training model predictions")
    train_data_df = ResultDB.get_data_static(
        model_dir / "train_predictions.hdf5", columns
    )

    console.log("Loading validation model predictions")
    val_data_df = ResultDB.get_data_static(model_dir / "val_predictions.hdf5", columns)

    # Compute the basin metrics
    console.log("Compute metrics")
    train_metrics = compute_basin_metrics(train_data_df, basin_dict, ims)
    val_metrics = compute_basin_metrics(val_data_df, basin_dict, ims)

    for cur_basin in basin_dict.keys():
        console.rule(cur_basin)
        console.log(
            pd.merge(
                train_metrics[cur_basin],
                val_metrics[cur_basin],
                left_index=True,
                right_index=True,
                suffixes=("_train", "_val"),
            )
        )

    return train_metrics, val_metrics


def wandb_log_metrics(
    wandb_run: object,
    general_metrics: pd.DataFrame,
    basin_metrics: Dict[str, pd.DataFrame],
    metrics: Sequence[str] = DEFAULT_METRICS,
):
    print("wtf")
