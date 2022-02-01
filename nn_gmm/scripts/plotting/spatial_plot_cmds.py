import time
from typing import Sequence, List, Optional
from pathlib import Path

import pygmt
import seaborn as sns
import ml_tools.utils
import numpy as np
import h5py
import pandas as pd
import typer
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import spatial
from scipy import interpolate
from qcore import geo

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


@app.command("metrics")
def gen_spatial_metric_plots(
    model_dir: Path,
    ims: List[str] = None,
    metrics: List[str] = None,
    n_procs: int = 4,
):
    """Creates spatial metrics plots"""
    ims = nn_gmm.GMM.load(model_dir).ims if not ims else ims
    metrics = nn_gmm.DEFAULT_METRICS if not metrics else metrics

    nn_gmm.gen_spatial_metric_plots(model_dir, ims, metrics=metrics, n_procs=n_procs)


@app.command("station-density")
def gen_station_density_plot(station_ffp: Path, output_ffp: Path, grid_size: int):
    # Read station file & drop duplicated locations
    station_df = pd.read_csv(
        station_ffp, index_col=2, sep="\s+", header=None, names=["lon", "lat"]
    )
    station_df.drop_duplicates(subset={"lat", "lon"}, inplace=True)

    # Create the land/water mask (which also acts as the aggregation grid)
    # There is probably a better way of doing this..
    land_mask = pygmt.grdlandmask(
        region="NZ",
        spacing=f"{grid_size}k/{grid_size}k",
        maskvalues=[0, 1],
        resolution="f"
        # region="NZ", spacing=f"1000+n/1000+n", maskvalues=[0, 1], resolution="f"
    ).T
    x1, x2 = np.meshgrid(land_mask.lon.values, land_mask.lat.values)

    # Convert to NZGD2000, which is in metres
    station_coords = geo.wgs_nztm2000x(
        np.stack((station_df.lon.values, station_df.lat.values), axis=1)
    )
    grid_coords = geo.wgs_nztm2000x(
        np.stack((x1.ravel(), x2.ravel()), axis=1)[land_mask.values.ravel() > 0, :]
    )

    # Create kd-tree and run lookup
    kd_tree = spatial.KDTree(station_coords)
    station_count = np.asarray(
        [
            len(cur_c)
            for cur_c in kd_tree.query_ball_point(
                grid_coords, r=grid_size * 1000, p=1, workers=-1
            )
        ]
    )

    # Create a high density (interpolated) grid for plotting
    grid = nn_gmm.create_grid(
        pd.DataFrame(
            np.concatenate(
                (geo.wgs_nztm2000x(grid_coords), station_count[:, np.newaxis]), axis=1
            ),
            columns=["lon", "lat", "count"],
        ),
        "count",
        interp_method="linear",
        # grid_spacing="200e/200e",
        grid_spacing="2k/2k",
    )

    # # Create fine grid for plotting
    # plt_land_mask = pygmt.grdlandmask(region="NZ", spacing="200e/200e", maskvalues=[0, 1], resolution="f").T
    # plt_grid = plt_land_mask.copy()
    # plt_grid.values = np.zeros(plt_grid.shape)
    # x1, x2 = np.meshgrid(plt_grid.lon.values, plt_grid.lat.values)
    # plt_grid_coords = geo.wgs_nztm2000x(np.stack((x1.ravel(), x2.ravel()), axis=1))
    #
    # # Interpolate available data onto meshgrid
    # interpolator = interpolate.CloughTocher2DInterpolator(
    #     grid_coords,
    #     station_count
    # )
    # plt_grid_values = interpolator(plt_grid_coords[:, 0], plt_grid_coords[:, 1])
    #
    # # Convert back to lat/lon
    # plt_grid.values = plt_grid_values.reshape(plt_grid.shape)
    # plt_grid.values[plt_land_mask.values.astype(bool)] = np.nan

    x1, x2 = np.meshgrid(grid.lon.values, grid.lat.values)


    fig = plt.figure(figsize=(16,10), dpi=1000)
    mask = ~np.isnan(grid.values)
    plt.scatter(x1[mask].ravel(), x2[mask].ravel(), c=grid.values[mask].ravel(), marker=",", s=0.1)

    fig.tight_layout()
    fig.savefig("/home/claudy/dev/work/tmp/station_density.png")


    fig = nn_gmm.gen_region_fig("Station density")

    nn_gmm.plot_grid(
        fig, grid, "hot", (0, 5, 5 / 10), ("white", "black"), reverse_cmap=True
    )

    fig.savefig(output_ffp, dpi=1200)

    print(f"wtf")


if __name__ == "__main__":
    app()
