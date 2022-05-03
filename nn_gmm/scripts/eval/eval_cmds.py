import time
from typing import Sequence, List, Optional
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


@app.command("write-predictions")
def write_train_val_predictions(data_dir: Path, model_dir: Path):
    """Write the training & validation predictions to hdf5 databases"""
    nn_gmm.write_train_val_predictions(data_dir, model_dir)


@app.command("metrics")
def train_val_metrics(model_dir: Path, save: bool = False):
    """Computes training & validation metrics"""
    nn_gmm.comp_train_val_metrics(model_dir, save=save)


@app.command("basin-metrics")
def train_val_basin_metrics(
    model_dir: Path, basin_dir: Path, save: bool = False, print: bool = True
):
    """Computes training & validation metrics"""
    nn_gmm.comp_train_val_basin_metrics(
        model_dir, basin_dir, save=save, print_metrics=print
    )


@app.command("spatial-metrics")
def train_val_spatial_metrics(model_dir: Path, save: bool = True):
    """Computes training & validation metrics"""
    nn_gmm.comp_train_val_spatial_metrics(model_dir, save=save)


@app.command("rrup-bin")
def gen_rrup_bin_plot(model_dir: Path, ims: List[str]):
    """Creates a Rrup based bin (mag, vs30)
    plot for the specified IM"""
    nn_gmm.gen_rrup_bin_plots(model_dir, ims)


@app.command("trend-rrup")
def trend_plot_rrup(
    model_dir: Path,
    im: str,
    source_config_ffp: Path,
    site_config_ffp: Path,
    site_source_config_ffp: Path,
):
    """Generates an Rrup trend plot"""
    nn_gmm.gen_rrup_trend_plots(
        model_dir,
        [im],
        ml_tools.utils.load_yaml(source_config_ffp),
        ml_tools.utils.load_yaml(site_config_ffp),
        ml_tools.utils.load_yaml(site_source_config_ffp),
    )


@app.command("residuals")
def residual_plots(model_dir: Path, ims: List[str] = None):
    """Generates residual plots"""
    nn_gmm.gen_residual_plots(model_dir, None if len(ims) == 0 else ims)


@app.command("spec-comp-bias-std")
def compare_spec_bias_std(
    output_ffp: Path, model_dirs: List[Path], val_only: bool = False
):
    """Generates a spectral bias/std plot for all given models"""
    db_ffps = [cur_model_dir / "val_predictions.hdf5" for cur_model_dir in model_dirs]
    if not val_only:
        db_ffps += [
            cur_model_dir / "train_predictions.hdf5" for cur_model_dir in model_dirs
        ]

    ims = nn_gmm.GMM.load(model_dirs[0]).ims

    res_plot_gen = nn_gmm.ResPlotGen()
    res_plot_gen.gen_spectral_bias_std_plot(db_ffps, ims, output_ffp)


@app.command("spec-comp-basin-bias-std")
def compare_spec_basin_bias_std(
    output_ffp: Path,
    model_dirs: List[Path],
    use_train: bool = False,
    use_val: bool = False,
    basin_ids: List[str] = None,
):
    """Creates a basin bias/std plot for the given models"""
    # Get the dbs
    assert (use_val or use_train) and not (
        use_val and use_train
    ), "Either use_train or use_val has to be set (but not both!)"

    res_plot_gen = nn_gmm.ResPlotGen()
    res_plot_gen.gen_spectral_basin_bias_std_plot(
        output_ffp,
        model_dirs,
        use_train=use_train,
        use_val=use_val,
        basin_ids=basin_ids,
    )


@app.command("spec-comp-loss")
def compare_spec_loss(output_ffp: Path, model_dirs: List[Path], val_only: bool = False):
    loss_dfs = [
        pd.read_csv(cur_model_dir / "loss.csv", index_col=0)
        for cur_model_dir in model_dirs
    ]
    run_ids = np.asarray([cur_model_dir.stem for cur_model_dir in model_dirs])
    ims = nn_gmm.GMM.load(model_dirs[0]).ims

    # Sort by run_id
    sort_ind = np.argsort(run_ids)
    run_ids = run_ids[sort_ind]
    loss_dfs = [loss_dfs[sort_ix] for sort_ix in sort_ind]

    nn_gmm.gen_spectral_loss_plot(loss_dfs, run_ids, ims, output_ffp, val_only=val_only)


@app.command("comp-basin-metrics-rrup-mag")
def compare_basin_metrics_mag_rrup(
    model_dirs: List[Path] = typer.Argument(
        ..., help="Directories of the models to compare"
    ),
    im: str = typer.Argument(...),
    metrics: List[str] = typer.Option(..., help="Metrics to compare"),
    regions: List[str] = typer.Option(..., help="Region/Basins to compare"),
    basin_dir: Path = typer.Argument(
        ..., help="Directorie that contains the basin definitions"
    ),
    output_dir: Path = typer.Argument(...),
    model_names: List[str] = typer.Option(
        None,
        help="Name of the models to use on the plot\n"
        "Has to be in the same order as model_dirs",
    ),
    val: bool = typer.Option(False, help="Generate for validation data"),
):
    """Creates a figure for each metric-region pair,
    showing the metrics trend (wrt. Magnitude and Rrup)
    for the region"""
    nn_gmm.gen_basin_comp_mag_rrup_plots(
        model_dirs,
        im,
        metrics,
        regions,
        basin_dir,
        output_dir,
        model_names=model_names,
        val=val,
    )


@app.command("comp-basin-metrics")
def compare_basin_metrics_matrix(
    model_dirs: List[Path] = typer.Argument(
        ..., help="The directories of the models to compare"
    ),
    output_dir: Path = typer.Argument(
        ..., help="Output directory path for the resulting plots"
    ),
    im: str = typer.Argument(...),
    metrics: List[str] = typer.Option(..., help="The comparison metrics"),
    model_names: List[str] = typer.Option(
        None,
        help="The names of the models to use on the plot\n"
        "If not specified then the run_id (without tags) is used",
    ),
    val: bool = typer.Option(False, help="Generate for validation data"),
):
    """Creates a matrix plot the metric
    for the specified models & available basins
    """
    for cur_metric in metrics:
        nn_gmm.gen_basin_metric_comp_matrix(
            [
                (cur_model_dir, model_names[ix] if model_names is not None else None)
                for ix, cur_model_dir in enumerate(model_dirs)
            ],
            im,
            cur_metric,
            output_dir / f"val_{cur_metric}.png"
            if val
            else output_dir / f"train_{cur_metric}.png",
            val=val,
        )


@app.command("avg-bias-std-spatial")
def avg_spatial_bias_std(
    model_dir: Path, ims: List[str] = None, metrics: List[str] = None, n_procs: int = 4
):
    ims = nn_gmm.GMM.load(model_dir).ims if not ims else ims
    metrics = None if not metrics else metrics

    nn_gmm.gen_spatial_metric_plots(model_dir, ims, n_procs=n_procs, metrics=metrics)


if __name__ == "__main__":
    app()
