import os
import time
from typing import Sequence, List, Optional, Dict, Union
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
    model_dir: Path, ims: List[str] = None, metrics: List[str] = None, n_procs: int = 4,
):
    """Creates spatial metrics plots"""
    ims = nn_gmm.GMM.load(model_dir).ims if not ims else ims
    metrics = nn_gmm.DEFAULT_METRICS if not metrics else metrics

    nn_gmm.gen_spatial_metric_plots(model_dir, ims, metrics=metrics, n_procs=n_procs)


@app.command("location-dependence")
def gen_location_dependence_plot(
    model_dir: Path, station_ffp: Path, ims: List[str] = None, mag: float = 7.0
):
    output_dir = model_dir / "plots" / "spatial_trend"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Merge custom constant features
    const_features = {**nn_gmm.DEFAULT_CONST_FEATURES, **{"mag": mag}}

    # Load station df
    station_df = pd.read_csv(
        station_ffp,
        delim_whitespace=True,
        index_col=2,
        header=None,
        names=["lon", "lat"],
    )

    # Get the base stations
    base_mask = np.char.startswith(station_df.index.values.astype(str), "0")
    lon_values = station_df.loc[base_mask, "lon"].values
    lat_values = station_df.loc[base_mask, "lat"].values
    n_loc = np.count_nonzero(base_mask)

    # Load the model
    gmm = nn_gmm.GMM.load(model_dir)
    ims = gmm.ims if len(ims) == 0 else ims

    if np.any(~np.isin(["lat", "lon"], gmm.features)) and np.any(
        ~np.isin(["X", "Y", "Z"], gmm.features)
    ):
        raise ValueError(
            "This functionality only works for models with location dependence!"
        )

    # Create the input dataframe
    input_df = pd.DataFrame.from_dict(
        {cur_key: [cur_value] * n_loc for cur_key, cur_value in const_features.items()}
    )

    if "lon" in gmm.features:
        input_df["lon"] = lon_values
        input_df["lat"] = lat_values
    else:
        input_df["X"] = np.cos(lat_values) * np.cos(lon_values)
        input_df["Y"] = np.cos(lat_values) * np.sin(lon_values)
        input_df["Z"] = np.sin(lat_values)

    # Get the model predictions
    result_df = gmm.predict(input_df)[0]

    # Add lon & lat values
    result_df["lon"] = lon_values
    result_df["lat"] = lat_values

    # Load the NZ map data
    nz_map_data = None
    if qcore_data_dir := os.environ.get("QCORE_DATA_DIR") is not None:
        nz_map_data = nn_gmm.NZMapData.load(Path(qcore_data_dir))

    # Create the figures
    for cur_im in ims:
        nn_gmm.console.print(f"Generating location-dependence plot for {cur_im}")
        cur_data = result_df.loc[:, [cur_im, "lon", "lat"]]

        fig = nn_gmm.gen_region_fig(
            title=f"{nn_gmm.get_im_name(cur_im)} Location Dependence",
            map_data=nz_map_data,
            plot_roads=False,
            plot_topo=False,
        )

        grid = nn_gmm.create_grid(cur_data, cur_im, interp_method="linear")
        cb_min, cb_max = (
            np.quantile(cur_data[cur_im], 0.05),
            np.quantile(cur_data[cur_im], 0.95),
        )
        nn_gmm.plot_grid(
            fig,
            grid,
            "polar",
            (cb_min, cb_max, np.abs(cb_max - cb_min) / 16),
            ("darkred", "darkblue"),
            reverse_cmap=False,
        )

        nn_gmm.console.print(f"Saving location-dependence plot for {cur_im}")
        fig.savefig(
            output_dir / f"{cur_im.replace('.', 'p')}_loc_dependence.png",
            dpi=900,
            anti_alias=True,
        )


@app.command("im-map")
def gen_im_map(data_dir: Path, rel_name: str, im: str, output_ffp: Path):
    record_ffp = next(data_dir.rglob(f"**/{rel_name.split('_')[0]}.tfrecord"))

    data_df = nn_gmm.load_tfrecord(
        str(record_ffp), nn_gmm.load_feature_details(record_ffp.parent)
    )
    data_df["rel"] = np.stack(
        np.char.rsplit(data_df.index.values.astype(str), "_", maxsplit=1)
    )[:, 0]

    data_df = data_df.loc[data_df.rel == rel_name]

    fig = nn_gmm.im_plot(data_df, im, rel_name, Path(os.environ.get("QCORE_DATA_DIR")))
    fig.savefig(str(output_ffp), dpi=1200)


@app.command("station-density")
def gen_station_density_plot(station_ffp: Path, output_ffp: Path, grid_size: int = 4):
    # Read station file & drop duplicated locations
    station_df = pd.read_csv(
        station_ffp, index_col=2, sep="\s+", header=None, names=["lon", "lat"]
    )
    station_df.drop_duplicates(subset={"lat", "lon"}, inplace=True)

    # Get the base stations
    base_stations_mask = np.char.startswith(station_df.index.values.astype(str), "0")

    # Convert to NZGD2000, which is in metres
    station_coords = geo.wgs_nztm2000x(
        np.stack((station_df.lon.values, station_df.lat.values), axis=1)
    )

    # Create kd-tree and run lookup
    kd_tree = spatial.KDTree(station_coords)
    station_count = np.asarray(
        [
            len(cur_c)
            for cur_c in kd_tree.query_ball_point(
                station_coords[base_stations_mask], r=grid_size * 1000, p=1, workers=-1
            )
        ]
    )

    # Create the plotting grid
    station_count_df = station_df.loc[base_stations_mask].copy()
    station_count_df["count"] = station_count
    grid = nn_gmm.create_grid(station_count_df, "count", interp_method="linear")

    # Generate the plot
    fig = nn_gmm.gen_region_fig("Station density")
    nn_gmm.plot_grid(
        fig, grid, "hot", (0, 50, 50 / 10), ("white", "black"), reverse_cmap=True
    )
    fig.savefig(output_ffp, dpi=1200)


if __name__ == "__main__":
    app()
