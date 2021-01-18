from pathlib import Path
from typing import List, Union, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .data import load_tfrecord, load_feature_details, load_dataset, load_basin_stations
from .model import GMM
from .plotting import plotting_utils as plt_utils


def get_realisation_residuals(
    data_dirs: Union[List[Path], Path],
    model: Union[GMM, Path],
    abs_residual: bool = True,
):
    """Computes realisation residuals for all realisations found in the
    specified data directories"""
    # Find all .tfrecord files
    if isinstance(data_dirs, list):
        tf_files = []
        for cur_dir in data_dirs:
            tf_files.extend(list(cur_dir.glob("*.tfrecord")))
    else:
        tf_files = list(data_dirs.glob("*.tfrecord"))

    # Load the model
    model = model if isinstance(model, GMM) else GMM.load(model)

    feature_details = (
        load_feature_details(data_dirs[0])
        if isinstance(data_dirs, list)
        else load_feature_details(data_dirs)
    )

    rel_res_dfs = []
    for ix, cur_tf_ffp in enumerate(tf_files):
        print(f"Processing {ix + 1}/{len(tf_files)}")

        cur_data = load_tfrecord(str(cur_tf_ffp), feature_details)
        cur_mean_df, _ = model.predict(cur_data)
        assert np.all(cur_mean_df.index == cur_data.index)

        ims = cur_mean_df.columns.values.astype(str)

        split_ids = np.stack(np.char.split(cur_mean_df.index.values.astype(str), "_"))
        cur_mean_df["realisation"] = np.char.add(
            np.char.add(split_ids[:, 0], "_"), split_ids[:, 1]
        )
        cur_data["realisation"] = cur_mean_df.realisation

        cur_data["station"] = cur_mean_df["station"] = split_ids[:, -1]

        cur_mean_df["fault"] = split_ids[:, 0]
        rel_fault = cur_mean_df.groupby("realisation").first()["fault"].to_frame()

        rel_mag = cur_data.groupby("realisation").first()["mag"]
        rel_n_stations = cur_data.groupby("realisation").count()["station"]
        rel_n_stations.name = "n_stations"

        cur_rel_res_dfs = []
        for im in ims:
            res = (
                cur_data.loc[:, im] - cur_mean_df.loc[:, im].apply(np.log)
            ).to_frame()
            if abs_residual:
                res[im] = res[im].apply(np.abs)

            res["realisation"] = cur_mean_df.realisation

            cur_rel_res_dfs.append(res.groupby("realisation").mean())

        cur_rel_res_df = pd.concat(cur_rel_res_dfs, axis=1)
        cur_rel_res_df = cur_rel_res_df.merge(
            rel_mag, left_index=True, right_index=True
        )
        cur_rel_res_df = cur_rel_res_df.merge(
            rel_fault, left_index=True, right_index=True
        )
        cur_rel_res_df = cur_rel_res_df.merge(
            rel_n_stations, left_index=True, right_index=True
        )
        rel_res_dfs.append(cur_rel_res_df)

    rel_res_df = pd.concat(rel_res_dfs)
    return rel_res_df


def get_loc_predictions(
    model: GMM, data_dir: Path, station_names: Union[List[str], np.ndarray]
):
    """Gets prediction for the locations of interest from the specified data directory"""
    ds = load_dataset(
        [data_dir],
        load_feature_details(data_dir),
        100_000,
        shuffle_buffer=None,
        block_size=10_000,
    )

    dfs = []
    for cur_batch in ds.as_numpy_iterator():
        cur_station_names = np.stack(np.char.split(cur_batch["id"].astype(str), "_"))[
            :, -1
        ]
        cur_mask = np.isin(cur_station_names, station_names)

        cur_df = pd.DataFrame(cur_batch, index=cur_batch["id"].astype(str)).loc[
            cur_mask
        ]
        cur_df["station"] = cur_station_names[cur_mask]
        dfs.append(cur_df)

    df = pd.concat(dfs)

    mean_df, std_df = model.predict(df)

    return mean_df, std_df, df


def get_basin_predictions(
    model: GMM, data_dir: Path, basins_stations: Dict[str, np.ndarray]
):
    """Gets model predictions for the specified basins"""
    stations = np.concatenate(
        [cur_stations for cur_stations in basins_stations.values()]
    )

    mean_df, std_df, df = get_loc_predictions(model, data_dir, stations)
    assert np.all(mean_df.index == df.index)

    df["basin"] = ""
    for cur_basin, cur_stations in basins_stations.items():
        cur_mask = np.isin(df.station, cur_stations)
        df.loc[cur_mask, "basin"] = cur_basin

    mean_df["station"], mean_df["basin"] = df.station, df.basin

    return mean_df, std_df, df


def run_location_eval(
    model: GMM,
    data_dir: Path,
    basin_dir: Path,
    fig_output_dir: Path = None,
    suffix: str = None,
):
    from .plotting.plotting_funcs import residual_scatter_hist_plot

    basin_dict = load_basin_stations(basin_dir)

    mean_df, _, df = get_basin_predictions(model, data_dir, basin_dict)

    ims = model.outputs
    suffix = "" if suffix is None else f"_{suffix}"

    # Compute mean absolute ln residual
    basin_mean_abs_ln_res = {}
    for cur_basin in basin_dict.keys():
        cur_mask = mean_df.basin == cur_basin
        cur_ln_res = df.loc[cur_mask, ims] - mean_df.loc[cur_mask, ims].apply(np.log)

        basin_mean_abs_ln_res[cur_basin] = cur_ln_res.apply(np.abs).mean(axis=0)

        # Create histogram plots of residual for each IM
        if fig_output_dir is not None:
            for ix, cur_im in enumerate(ims):
                plt.figure()
                plt.hist(cur_ln_res[cur_im], bins=25)
                plt.xlabel(cur_im)
                plt.text(
                    0.99,
                    0.99,
                    f"Mean abs ln res: {basin_mean_abs_ln_res[cur_basin][cur_im]:.4f}",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=plt.gca().transAxes,
                )
                plt.xlim((-1.75, 1.75))

                plt.tight_layout()
                plt.savefig(fig_output_dir / f"{cur_basin}_{cur_im.replace('.', 'p')}_ln_res{suffix}.png")
                plt.close()

    return basin_mean_abs_ln_res
