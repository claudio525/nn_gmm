import os
import multiprocessing as mp
from typing import List, Union, Tuple
from pathlib import Path
from importlib import reload

import numpy as np
import pandas as pd
import typer
import tensorflow as tf
from scipy import spatial
from qcore import geo, nhm

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
import ml_tools

app = typer.Typer()


@app.command("faults-plot")
def gen_faults_plot(
    output_dir: Path,
    source_rel_data_dir: Path,
    nshm_ffp: Path,
    map_data_ffp: Path,
    val_events_ffp: Path = None,
    show_hypo: bool = False,
):
    """Generates two plots (or one if val_events_ffp is not specified)
    showing all the fault traces (future events) and historic events (circles)

    Note, is not fully generalised, i.e. only works for
    small, moderate and cybershake data
    """
    # Load the data
    nhm_data = nhm.load_nhm(str(nshm_ffp))

    cybershake_df = pd.read_csv(
        source_rel_data_dir / "cybershake_v20p4_200.csv", index_col=0
    )
    cybershake_df["fault"] = np.stack(
        np.char.split(cybershake_df.index.values.astype(str), "_", maxsplit=1), axis=1
    )[0, :]
    cybershake_df["historic"] = False

    small_df = pd.read_csv(source_rel_data_dir / "val_small.csv", index_col=0)
    small_df["fault"] = small_df.index.values.astype(str)
    small_df["historic"] = True

    mod_df = pd.read_csv(source_rel_data_dir / "val_moderate.csv", index_col=0)
    mod_df["fault"] = mod_df.index.values.astype(str)
    mod_df["historic"] = True

    rupture_df = pd.concat((cybershake_df, small_df, mod_df), axis=0)

    # Plotting data
    map_data = nn_gmm.NZMapData.load(map_data_ffp)

    if val_events_ffp is not None:
        val_faults = ml_tools.utils.load_txt(val_events_ffp)
        val_mask = np.isin(rupture_df.fault, val_faults)
        val_faults = [
            cur_fault
            for cur_name, cur_fault in nhm_data.items()
            if cur_name in val_faults
        ]
        train_faults = [
            cur_fault
            for cur_name, cur_fault in nhm_data.items()
            if cur_name not in val_faults
        ]

        fig = nn_gmm.faults_plot(
            rupture_df.loc[~val_mask],
            train_faults,
            map_data=map_data,
            title="Training Faults",
            show_hypo=show_hypo,
        )
        fig.savefig(
            output_dir / f"train_faults.png", dpi=900, anti_alias=True,
        )

        fig = nn_gmm.faults_plot(
            rupture_df.loc[val_mask],
            val_faults,
            map_data=map_data,
            title="Validation Faults",
            show_hypo=show_hypo,
        )
        fig.savefig(
            output_dir / f"train_faults.png", dpi=900, anti_alias=True,
        )

    else:
        faults = [cur_fault for cur_name, cur_fault in nhm_data.items()]

        fig = nn_gmm.faults_plot(
            rupture_df, faults, map_data=map_data, title="Faults", show_hypo=show_hypo
        )
        fig.savefig(
            output_dir / f"faults.png", dpi=900, anti_alias=True,
        )


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


@app.command("sim-im-map")
def gen_im_map(
    data_dir: Path = typer.Argument(
        ..., help="The tfrecord data directory that contains the fault file to plot"
    ),
    fault: str = typer.Argument(..., help="Name of the fault to plot"),
    ims: List[str] = typer.Argument(..., help="IM to plot"),
    output_dir: Path = typer.Argument(..., help="Output directory",),
    rel_name: str = typer.Option(
        None,
        help="Realisation to plot, if not given then all realisation for the specified fault are plotted",
    ),
    nhm_ffp: Path = typer.Option(
        None, help="Path to the NHM file, to allow showing the fault trace"
    ),
    qcore_data_dir: Path = typer.Option(
        None, help="Path to the qcore data dir. Required for road & topo mapping."
    ),
    n_procs: int = typer.Option(
        1,
        help="Number of processes to use, only matters when plotting multiple realisations",
    ),
):
    """Generates simulation based IM maps from a tfrecord file"""
    record_ffp = next(data_dir.rglob(f"**/{fault}.tfrecord"))

    # Load the data
    data_df = nn_gmm.load_tfrecord(
        str(record_ffp), nn_gmm.load_feature_details(record_ffp.parent)
    )
    data_df["rel"] = np.stack(
        np.char.rsplit(data_df.index.values.astype(str), "_", maxsplit=1)
    )[:, 0]

    nn_gmm.im_plots(
        data_df,
        fault,
        ims,
        output_dir,
        rel_name=rel_name,
        nhm_ffp=nhm_ffp,
        qcore_data_dir=qcore_data_dir,
        n_procs=n_procs,
    )


@app.command("model-im-map")
def gen_model_im_map(
    model_dir: Path = typer.Argument(
        ..., help="The model directory. Must contain the prediction DBs"
    ),
    source_params_dir: Path = typer.Argument(
        ..., help="Path to the source parameters directory"
    ),
    fault: str = typer.Argument(..., help="Name of the fault to plot"),
    ims: List[str] = typer.Argument(..., help="IM to plot"),
    output_dir: Path = typer.Option(None, help="Output directory",),
    rel_name: str = typer.Option(
        None,
        help="Realisation to plot, if not given then all realisation for the specified fault are plotted",
    ),
    nhm_ffp: Path = typer.Option(
        None, help="Path to the NHM file, to allow showing the fault trace"
    ),
    qcore_data_dir: Path = typer.Option(
        None, help="Path to the qcore data dir. Required for road & topo mapping."
    ),
    cb_min_values: List[float] = typer.Option(
        None, help="List of minimum color-bar (and colormap) values, same order as ims"
    ),
    cb_max_values: List[float] = typer.Option(
        None, help="List of maximum color-bar (and colormap) value, same order as ims"
    ),
    n_procs: int = typer.Option(
        1,
        help="Number of processes to use, only matters when plotting multiple realisations",
    ),
):
    """Generates IM maps for the specified using the given model"""
    # Load source params
    rel_df = nn_gmm.load_dfs(
        list(source_params_dir.glob("*.csv")), index_col="realisation"
    )
    rel_df["fault"] = [
        split_list[0]
        for split_list in np.char.split(rel_df.index.values.astype(str), "_")
    ]
    rel_df = rel_df.loc[rel_df.fault == fault]

    console.print("Loading predictions")
    im_keys = [f"{im}_est" for im in ims]
    columns = im_keys + ["fault", "lat", "lon", "hlat", "hlon"]
    train_data = nn_gmm.ResultDB.get_data_static(
        model_dir / "train_predictions.hdf5", columns
    )
    val_data = nn_gmm.ResultDB.get_data_static(
        model_dir / "val_predictions.hdf5", columns
    )

    # Prepare data frame for plotting
    data_df = (
        train_data.loc[train_data.fault == fault].copy()
        if fault in train_data.fault.values
        else val_data.loc[val_data.fault == fault].copy()
    )
    del train_data, val_data
    data_df["rel"] = np.stack(
        np.char.rsplit(data_df.index.values.astype(str), "_", maxsplit=1)
    )[:, 0]

    # Rename im columns
    data_df.rename(
        columns={cur_im_key: cur_im for cur_im_key, cur_im in zip(im_keys, ims)},
        inplace=True,
    )

    # Merge with source/realisation params
    data_df = data_df.merge(
        rel_df, left_on="rel", right_index=True, suffixes=("", "_right")
    )
    data_df.drop(columns=["fault_right"], inplace=True)

    if output_dir is None:
        output_dir = model_dir / "plots" / "im_plots"
        output_dir.mkdir(exist_ok=True)

    cb_limits = None
    if len(cb_min_values) > 0 and len(cb_max_values) > 0:
        cb_limits = {
            cur_im: (cur_min, cur_max)
            for cur_min, cur_max, cur_im in zip(cb_min_values, cb_max_values, ims)
        }

    console.print(f"Generating maps")
    nn_gmm.im_plots(
        data_df,
        fault,
        ims,
        output_dir,
        rel_name=rel_name,
        nhm_ffp=nhm_ffp,
        qcore_data_dir=qcore_data_dir,
        cb_limits_dict=cb_limits,
        n_procs=n_procs,
    )


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
