import time
from typing import Sequence, List
from pathlib import Path

import seaborn as sns
import ml_tools.utils
import numpy as np
import h5py
import pandas as pd
import typer
import tensorflow as tf

# Grow the GPU memory usage as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import nn_gmm

app = typer.Typer()


@app.command("write")
def write_train_val_predictions(data_dir: Path, model_dir: Path):
    """Write the training & validation predictions to hdf5 databases"""
    nn_gmm.write_train_val_predictions(data_dir, model_dir)


@app.command("metrics")
def train_val_metrics(model_dir: Path, save: bool = False):
    """Computes training & validation metrics"""
    nn_gmm.train_val_metrics(model_dir, save=save)


@app.command("rrup")
def gen_rrup_bin_plot(model_dir: Path, ims: List[str]):
    """Creates a Rrup based bin (mag, vs30)
    plot for the specified IM"""
    nn_gmm.gen_rrup_bin_plot(model_dir, ims)


@app.command("basin")
def basin_eval(model_dir: Path, basin_dir: Path):
    """Computes basin metrics"""
    nn_gmm.train_val_basin_metrics(model_dir, basin_dir)


@app.command("trend")
def trend_plot(model_dir: Path, im: str):
    """Generates an Rrup trend plot"""
    nn_gmm.gen_rrup_trend_plot(model_dir, [im])


@app.command("residuals")
def residual_plots(model_dir: Path, ims: List[str] = None):
    """Generates residual plots"""
    nn_gmm.gen_residual_plots(model_dir, None if len(ims) == 0 else ims)


@app.command("spec-comp-residuals")
def compare_residuals(output_ffp: Path, model_dirs: List[Path], val_only: bool = False):
    """Generates a spectral bias/std plot for all given models"""
    db_ffps = [
        cur_model_dir / "val_predictions.hdf5" for cur_model_dir in model_dirs
    ]
    if not val_only:
        db_ffps += [cur_model_dir / "train_predictions.hdf5" for cur_model_dir in model_dirs]

    ims = nn_gmm.GMM.load(model_dirs[0]).ims

    res_plot_gen = nn_gmm.ResPlotGen()
    res_plot_gen.gen_spectral_bias_std_plot(db_ffps, ims, output_ffp)


@app.command("spec-comp-loss")
def compare_loss(output_ffp: Path, model_dirs: List[Path], val_only: bool = False):
    loss_dfs = [pd.read_csv(cur_model_dir / "loss.csv", index_col=0) for cur_model_dir in model_dirs]
    run_ids = np.asarray([cur_model_dir.stem for cur_model_dir in model_dirs])
    ims = nn_gmm.GMM.load(model_dirs[0]).ims

    # Sort by run_id
    sort_ind = np.argsort(run_ids)
    run_ids = run_ids[sort_ind]
    loss_dfs = [loss_dfs[sort_ix] for sort_ix in sort_ind]

    nn_gmm.gen_spectral_loss_plot(loss_dfs, run_ids, ims, output_ffp, val_only=val_only)

if __name__ == "__main__":
    app()
