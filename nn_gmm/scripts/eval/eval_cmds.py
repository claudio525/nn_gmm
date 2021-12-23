import time
from typing import Sequence, List
from pathlib import Path

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
from nn_gmm import console

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
def gen_rrup_bin_plot(model_dir: Path, im: str):
    """Creates a Rrup based plot for the specified IM"""
    nn_gmm.gen_rrup_bin_plot(model_dir, im)


@app.command("basin")
def basin_eval(model_dir: Path, basin_dir: Path):
    """Computes basin metrics"""
    nn_gmm.train_val_basin_metrics(model_dir, basin_dir)


@app.command("trend")
def trend_plot(model_dir: Path, im: str):
    """Generates an Rrup trend plot"""
    nn_gmm.gen_rrup_trend_plot(model_dir, im)


@app.command("residuals")
def residual_plots(model_dir: Path, ims: List[str] = None):
    """Generates residual plots"""
    nn_gmm.gen_residual_plots(model_dir, None if len(ims) == 0 else ims)



if __name__ == "__main__":
    app()
