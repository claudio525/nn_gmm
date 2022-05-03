"""Script for creating the tfrecord files

Disable GPU for this
"""

import time

import pickle
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import spatial

from qcore import geo
import nn_gmm
from nn_gmm import console

IMs = np.asarray(
    [
        "AI",
        "CAV",
        "Ds575",
        "Ds595",
        "MMI",
        "PGA",
        "PGV",
        "pSA_0.01",
        "pSA_0.02",
        "pSA_0.03",
        "pSA_0.04",
        "pSA_0.05",
        "pSA_0.075",
        "pSA_0.1",
        "pSA_0.12",
        "pSA_0.15",
        "pSA_0.17",
        "pSA_0.2",
        "pSA_0.25",
        "pSA_0.3",
        "pSA_0.4",
        "pSA_0.5",
        "pSA_0.6",
        "pSA_0.7",
        "pSA_0.75",
        "pSA_0.8",
        "pSA_0.9",
        "pSA_1.0",
        "pSA_1.25",
        "pSA_1.5",
        "pSA_2.0",
        "pSA_2.5",
        "pSA_3.0",
        "pSA_4.0",
        "pSA_5.0",
        "pSA_6.0",
        "pSA_7.5",
        "pSA_10.0",
    ]
)

BIN_RRUP_MIN, BIN_RRUP_MAX = 0, 395


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_features(values: np.ndarray):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _serialize(
    ix_1, input_row, ix_2, im_df_row, ix_3, site_source_row, sample_weight: float
):
    assert ix_1 == ix_2 and ix_1 == ix_3

    features = {
        **{key: _float_feature(value) for key, value in input_row.items()},
        **{key: _float_feature(value) for key, value in im_df_row.items()},
    }
    features["id"] = _bytes_feature(str.encode(ix_1))
    features["sample_weight"] = _float_feature(sample_weight)
    features = {
        **features,
        **{
            key: _bytes_feature(str.encode(value))
            for key, value in site_source_row.items()
        },
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def serialize(
    input_df: pd.DataFrame,
    im_df: pd.DataFrame,
    site_source_ids: pd.DataFrame,
    sample_weights: pd.Series,
    n_procs: int = 8,
):
    """Serializes training data (features & labels) into the tf.train.Example format"""
    ser_examples = []

    if input_df.shape[0] < 1000 or n_procs == 1:
        for (
            (ix_1, cur_input_row),
            (ix_2, cur_im_df_row),
            (ix_3, cur_site_source_row),
        ) in zip(input_df.iterrows(), im_df.iterrows(), site_source_ids.iterrows()):
            ser_examples.append(
                _serialize(
                    ix_1,
                    cur_input_row,
                    ix_2,
                    cur_im_df_row,
                    ix_3,
                    cur_site_source_row,
                    sample_weights[ix_1],
                )
            )
    else:
        with mp.Pool(processes=n_procs) as pool:
            ser_examples = pool.starmap(
                _serialize,
                [
                    (
                        ix_1,
                        cur_input_row,
                        ix_2,
                        cur_im_df_row,
                        ix_3,
                        cur_site_source_row,
                        sample_weights[ix_1],
                    )
                    for (ix_1, cur_input_row), (ix_2, cur_im_df_row), (
                        ix_3,
                        cur_site_source_row,
                    ) in zip(
                        input_df.iterrows(),
                        im_df.iterrows(),
                        site_source_ids.iterrows(),
                    )
                ],
            )

    return ser_examples


def _compute_site_sample_count(
    source: str, site_df: pd.DataFrame, imdb_ffps: Sequence[Path]
):
    """Gets the number of samples (i.e. ruptures) for the specified site"""
    site_counter = pd.Series(
        index=site_df.index, name=source, data=np.zeros(site_df.shape[0], dtype=int)
    )

    # Get the IM data (most reliable way to get datapoints)
    im_df = nn_gmm.load_fault_im_df(source, imdb_ffps)
    if im_df is None:
        return None

    id_split = np.stack(
        np.char.rsplit(im_df.index.values.astype(str), "_", maxsplit=1), axis=0
    )

    n_ruptures = np.unique(id_split[:, 0]).size
    sites = id_split[:, -1]

    site_counter.loc[sites[np.isin(sites, site_df.index)]] += n_ruptures

    return site_counter


def gen_tf_records(
    sources: np.ndarray,
    rel_df: pd.DataFrame,
    site_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    directivity_df: pd.DataFrame,
    im_db_ffps: List[Path],
    output_dir: Path,
    tect_type_one_hot_dict: Dict,
    n_procs: int = 8,
    fault_density_weights: pd.Series = None,
    station_density_weights: pd.Series = None,
):
    """Generates tfrecord files using the tf.train.Example protocol,
    one file is generated per event
    """
    console.print("Generating tfrecord files")
    n_sources = sources.size
    for ix, cur_source in enumerate(sources):
        console.print(f"Processing {ix + 1}/{n_sources}")
        cur_output_ffp = output_dir / f"{cur_source}.tfrecord"
        if cur_output_ffp.exists():
            console.print(f"Skipping source {cur_source} as output tfrecord already exists")
            continue

        cur_im_df = nn_gmm.load_fault_im_df(cur_source, im_db_ffps)
        if cur_im_df is None:
            console.print(f"No IM data found for source {cur_source}, skipping.")
            continue

        cur_im_df.sort_index(inplace=True)

        # Create sample combinations
        cur_input_df = nn_gmm.create_sample_comb(cur_im_df)
        cur_input_df["id"] = cur_input_df.index.values.astype(str)

        # Merge with realisation source parameters
        cur_input_df = pd.merge(
            cur_input_df,
            rel_df,
            how="inner",
            left_on="realisation",
            right_index=True,
            suffixes=(None, "_rel_df"),
        )

        # Merge with site parameters
        cur_input_df = pd.merge(
            cur_input_df, site_df, how="inner", left_on="site", right_index=True
        )

        # Merge with directivity parameters
        cur_input_df = pd.merge(
            cur_input_df,
            directivity_df.loc[:, ["T", "U", "S", "D"]],
            how="inner",
            left_index=True,
            right_index=True,
        )

        # Merge with distance parameters
        cur_input_df = pd.merge(
            cur_input_df,
            distance_df,
            how="inner",
            left_on=["source", "site"],
            right_on=["source", "site"],
        )

        # Tidy up
        cur_input_df.set_index("id", inplace=True)
        cur_input_df.sort_index(inplace=True)

        # Get source & site ids
        cur_site_source_ids = cur_input_df.loc[:, ["source", "site"]]

        # Drop non-feature columns
        cur_input_df = cur_input_df.drop(
            columns=[
                "realisation",
                "source_rel_df",
                "site",
                "source",
                "rtvz",
                "fault_id",
            ]
        )
        cur_input_df = nn_gmm.apply_one_hot_enc(
            cur_input_df, "tect_type", tect_type_one_hot_dict
        )

        # Update cur_im_df for missing/excluded sites
        cur_im_df = cur_im_df.loc[cur_input_df.index.values]

        # Interpolate for any extra periods
        cur_im_df = nn_gmm.interpolate_pSA_periods(cur_im_df, IMs)

        # Compute the sample weights
        sample_weights = pd.Series(
            index=cur_site_source_ids.index,
            data=np.ones(cur_site_source_ids.shape[0], dtype=float),
        )
        if fault_density_weights is not None:
            sample_weights *= fault_density_weights.loc[
                cur_site_source_ids.site.values
            ].values
        if station_density_weights is not None:
            sample_weights *= station_density_weights.loc[cur_site_source_ids.site.values].values

        # Serialize
        assert np.all(
            cur_input_df.index.values.astype(str) == cur_im_df.index.values.astype(str)
        )
        examples = serialize(
            cur_input_df,
            cur_im_df[IMs],
            cur_site_source_ids,
            sample_weights,
            n_procs=n_procs,
        )

        feature_details_ffp = output_dir / "feature_details.pickle"
        if ix == 0 or not feature_details_ffp.exists():
            console.print(f"Writing feature details")
            feature_description = {
                **{
                    col: tf.io.FixedLenFeature([], tf.float32)
                    for col in cur_input_df.columns.values.astype(str)
                },
                **{
                    col: tf.io.FixedLenFeature([], tf.float32)
                    for col in cur_im_df.columns.values.astype(str)
                },
                **{
                    col: tf.io.FixedLenFeature([], tf.string)
                    for col in cur_site_source_ids.columns.values.astype(str)
                },
            }
            feature_description = {
                **feature_description,
                **{
                    "id": tf.io.FixedLenFeature([], tf.string),
                    "sample_weight": tf.io.FixedLenFeature([], tf.float32),
                },
            }
            with open(str(feature_details_ffp), "wb") as f:
                pickle.dump(feature_description, f)

        with tf.io.TFRecordWriter(str(cur_output_ffp)) as writer:
            for example in examples:
                writer.write(example)


def main(
    site_params_dir: Path,
    distance_dir: Path,
    directivity: Path,
    source_params_dir: Path,
    im_db_dir: Path,
    output_dir: str,
    base_grid_only: bool = False,
    val_events_ffp: Path = None,
    n_procs: int = 8,
    use_fault_density_weights: bool = False,
    use_site_density_weights: bool = False,
):
    output_dir = Path(output_dir)

    # Load Site params
    console.print("Loading site params")
    site_df = nn_gmm.load_dfs(list(site_params_dir.glob("*.csv")))

    # Perform X, Y, Z coordinate transform as per
    # https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature
    # x = cos(lat) * cos(lon)
    # y = cos(lat) * sin(lon),
    # z = sin(lat)
    site_df["X"] = np.cos(site_df.lat) * np.cos(site_df.lon)
    site_df["Y"] = np.cos(site_df.lat) * np.sin(site_df.lon)
    site_df["Z"] = np.sin(site_df.lat)

    # Check & Drop duplicates
    n_unique_stations = np.unique(site_df.station.values.astype(str)).shape[0]
    site_df = site_df.drop_duplicates()
    assert n_unique_stations == site_df.shape[0]
    site_df.set_index("station", inplace=True)
    site_df.drop_duplicates(subset={"lat", "lon"}, inplace=True)

    # Only use base grid (and real) stations
    if base_grid_only:
        station_ids = site_df.index.values.astype(str)
        mask = (
            np.char.startswith(station_ids, "1")
            | np.char.startswith(station_ids, "2")
            | np.char.startswith(station_ids, "3")
            | np.char.startswith(station_ids, "4")
        )
        site_df = site_df.loc[~mask]

    # Load site-source params
    console.print("Loading distance params")
    distance_db_ffps = list(distance_dir.glob("*.db"))
    distance_df = nn_gmm.load_distance_df(site_df, distance_db_ffps, n_procs=n_procs)

    console.print("Loading directivity params")
    directivity_db_ffps = list(directivity.glob("*.db"))
    directivity_df = nn_gmm.load_directivity_df(directivity_db_ffps, n_procs=n_procs)
    assert (
        np.unique(directivity_df.index.values.astype(str)).shape[0]
        == directivity_df.shape[0]
    )

    # Load source params
    console.print("Loading realisation params")
    rel_df = nn_gmm.load_dfs(
        list(source_params_dir.glob("*.csv")), index_col="realisation"
    )
    rel_df["source"] = [
        split_list[0]
        for split_list in np.char.split(rel_df.index.values.astype(str), "_")
    ]

    # Split events/sources into train/validation data
    im_db_ffps = list(im_db_dir.glob("*.h5"))
    if val_events_ffp is not None:
        all_sources = np.unique(rel_df.source.values.astype(str))
        with open(val_events_ffp, "r") as f:
            val_sources = np.asarray([line.strip() for line in f.readlines()])
        train_sources = all_sources[~np.isin(all_sources, val_sources)]

        fault_density_weights = None
        if use_fault_density_weights:
            console.print("Computing fault density weights for each site")
            if n_procs == 1:
                site_counters = []
                for ix, cur_source in enumerate(train_sources):
                    site_counters.append(
                        _compute_site_sample_count(cur_source, site_df, im_db_ffps)
                    )
            else:
                with mp.Pool(n_procs) as p:
                    site_counters = p.starmap(
                        _compute_site_sample_count,
                        [
                            (cur_source, site_df, im_db_ffps)
                            for cur_source in train_sources
                        ],
                    )

            fault_density_weights = 1 / pd.concat(site_counters, axis=1).sum(axis=1)
            fault_density_weights = fault_density_weights.loc[
                ~np.isinf(fault_density_weights)
            ]
            assert np.all((fault_density_weights <= 1.0) & (fault_density_weights > 0.0))

            # Normalise to sum to the number of stations
            fault_density_weights = fault_density_weights * (fault_density_weights.shape[0] / fault_density_weights.sum())

        station_density_weights = None
        if use_site_density_weights:
            console.print(f"Computing station density weights for each site")
            site_nztm200_coords = geo.wgs_nztm2000x(np.stack((site_df.lon.values, site_df.lat.values), axis=1))

            # Create kd-tree and run lookup
            kd_tree = spatial.KDTree(site_nztm200_coords)
            station_count = np.asarray(
                [
                    len(cur_c)
                    for cur_c in kd_tree.query_ball_point(
                    site_nztm200_coords, r=3.99 * 1000, p=2,
                    workers=-1
                )
                ]
            )

            station_density_weights = 1 / pd.Series(index=site_df.index, data=station_count)
            # Normalise so that sum of weights == number of stations (not really needed tbh)
            station_density_weights = station_density_weights * site_df.shape[0] / station_density_weights.sum()

        # Training dataset
        gen_tf_records(
            train_sources,
            rel_df,
            site_df,
            distance_df,
            directivity_df,
            im_db_ffps,
            output_dir / "train",
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
            n_procs=n_procs,
            fault_density_weights=fault_density_weights,
            station_density_weights=station_density_weights
        )

        # Validation dataset
        gen_tf_records(
            val_sources,
            rel_df,
            site_df,
            distance_df,
            directivity_df,
            im_db_ffps,
            output_dir / "val",
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
            n_procs=n_procs,
        )
    else:
        if use_fault_density_weights:
            raise NotImplementedError()

        gen_tf_records(
            rel_df.index.values.astype(str),
            rel_df,
            site_df,
            distance_df,
            directivity_df,
            im_db_ffps,
            output_dir,
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
            n_procs=n_procs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "site_params_dir", type=str, help="The path to the site params dir"
    )
    parser.add_argument(
        "distance_dir", type=str, help="The path to the distance site-source dir"
    )
    parser.add_argument(
        "site_source_dir", type=str, help="The path to the site-source dir"
    )
    parser.add_argument(
        "source_params_dir",
        type=str,
        help="The path to the source params dir",
    )
    parser.add_argument("im_db_dir", type=str, help="The path to the IM labels dbs dir")
    parser.add_argument("output_dir", type=str, help="Path of the output directory")
    parser.add_argument(
        "--base_grid_only",
        action="store_true",
        default=False,
        help="If set, only the uniform base grid (and real) stations are used",
    )
    parser.add_argument(
        "--val_events_ffp",
        type=str,
        help="Path to a text file that is a list of validation events (one per line)",
    )
    parser.add_argument(
        "--n_procs", type=int, help="Number of processes to use", default=4
    )
    parser.add_argument(
        "--fault_density_weights",
        action="store_true",
        help="If set then samples are weighted based on number "
        "of other datapoints (i.e. faults/realisations) at a given site",
    )
    parser.add_argument(
        "--station_density_weights",
        action="store_true",
        help="If set then sites are weight based "
        "on the number of other stations in their vicinity (radius of 3.8km)",
    )
    args = parser.parse_args()

    main(
        nn_gmm.to_path(args.site_params_dir),
        nn_gmm.to_path(args.distance_dir),
        nn_gmm.to_path(args.site_source_dir),
        nn_gmm.to_path(args.source_params_dir),
        nn_gmm.to_path(args.im_db_dir),
        nn_gmm.to_path(args.output_dir),
        base_grid_only=args.base_grid_only,
        n_procs=args.n_procs,
        val_events_ffp=nn_gmm.to_path(args.val_events_ffp),
        use_fault_density_weights=args.fault_density_weights,
        use_site_density_weights=args.station_density_weights,
    )
