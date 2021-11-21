import pickle
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf

import nn_gmm

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


def _serialize(ix_1, cur_input_row, ix_2, cur_im_df_row, sample_weight: float):
    assert ix_1 == ix_2

    features = {
        **{key: _float_feature(value) for key, value in cur_input_row.items()},
        **{key: _float_feature(value) for key, value in cur_im_df_row.items()},
    }
    features["id"] = _bytes_feature(str.encode(ix_1))
    features["sample_weight"] = _float_feature(sample_weight)

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def serialize(input_df: pd.DataFrame, im_df: pd.DataFrame, sample_weights_df: pd.Series, n_procs: int = 8):
    """Serializes training data (features & labels) into the tf.train.Example format"""
    ser_examples = []

    if input_df.shape[0] < 1000 or n_procs == 1:
        for (ix_1, cur_input_row), (ix_2, cur_im_df_row) in zip(
            input_df.iterrows(), im_df.iterrows()
        ):
            ser_examples.append(_serialize(ix_1, cur_input_row, ix_2, cur_im_df_row, sample_weights_df.loc[ix_1]))
    else:
        with mp.Pool(processes=n_procs) as pool:
            ser_examples = pool.starmap(
                _serialize,
                [
                    (ix_1, cur_input_row, ix_2, cur_im_df_row, sample_weights_df.loc[ix_1])
                    for (ix_1, cur_input_row), (ix_2, cur_im_df_row) in zip(
                        input_df.iterrows(), im_df.iterrows()
                    )
                ],
            )

    return ser_examples


def gen_tf_records(
    sources: np.ndarray,
    rel_df: pd.DataFrame,
    site_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    site_source_df: pd.DataFrame,
    im_db_ffps: List[Path],
    output_dir: Path,
    tect_type_one_hot_dict: Dict,
    rrup_mag_weighting_data: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
    n_procs: int = 8,
):
    """Generates tfrecord files using the tf.train.Example protocol,
    one file is generated per event
    """
    print("Generating tfrecord files")
    total_sample_weights = 0
    for ix, cur_source in enumerate(sources):
        print(f"Processing {ix + 1}/{sources.size}")
        cur_output_ffp = output_dir / f"{cur_source}.tfrecord"
        if cur_output_ffp.exists():
            print(f"Skipping source {cur_source} as output tfrecord already exists")
            continue

        cur_im_df = nn_gmm.load_fault_im_df(cur_source, im_db_ffps)
        if cur_im_df is None:
            print(f"No IM data found for source {cur_source}, skipping.")
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

        # Merge with site-source parameters
        cur_input_df = pd.merge(
            cur_input_df,
            site_source_df.loc[:, ["theta", "s"]],
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
        cur_input_df.set_index("id", inplace=True)
        cur_input_df.sort_index(inplace=True)
        cur_input_df = cur_input_df.drop(
            columns=["realisation", "source", "source_rel_df", "site", "rtvz"]
        )
        # cur_input_df = cur_input_df.drop(columns=["source", "site"])
        cur_input_df = nn_gmm.apply_one_hot_enc(
            cur_input_df, "tect_type", tect_type_one_hot_dict
        )

        cur_im_df = nn_gmm.interpolate_pSA_periods(cur_im_df, IMs)

        # Get the sample weights
        sample_weights_df = pd.Series(index=cur_input_df.index, data=np.ones(cur_input_df.shape[0], dtype=float))
        if rrup_mag_weighting_data is not None:
            bin_weights, mag_bins, rrup_bins = rrup_mag_weighting_data

            mag_diff = mag_bins - cur_input_df.mag.values[:, None]
            mag_diff[mag_diff >= 0] = -np.inf
            mag_bin_ind = mag_diff.argmax(axis=1)

            rrup_diff = rrup_bins - cur_input_df.rrup.values[:, None]
            rrup_diff[rrup_diff >= 0] = -np.inf
            rrup_bin_ind = rrup_diff.argmax(axis=1)

            sample_weights = bin_weights[rrup_bin_ind, mag_bin_ind]
            total_sample_weights += sample_weights.sum()

            sample_weights_df = pd.Series(index=cur_input_df.index, data=sample_weights)

        assert np.all(
            cur_input_df.index.values.astype(str) == cur_im_df.index.values.astype(str)
        )
        examples = serialize(cur_input_df, cur_im_df[IMs], sample_weights_df, n_procs=n_procs)

        if ix == 0:
            print(f"Writing feature details")
            feature_description = {
                **{
                    col: tf.io.FixedLenFeature([], tf.float32)
                    for col in cur_input_df.columns.values.astype(str)
                },
                **{
                    col: tf.io.FixedLenFeature([], tf.float32)
                    for col in cur_im_df.columns.values.astype(str)
                },
            }
            feature_description = {
                **feature_description,
                **{"id": tf.io.FixedLenFeature([], tf.string),
                   "sample_weight": tf.io.FixedLenFeature([], tf.float32)},
            }
            with open(str(output_dir / "feature_details.pickle"), "wb") as f:
                pickle.dump(feature_description, f)

        with tf.io.TFRecordWriter(str(cur_output_ffp)) as writer:
            for example in examples:
                writer.write(example)

    if rrup_mag_weighting_data is not None:
        if not np.isclose(total_sample_weights, 1.0):
            print("Sample weights don't add up to 1.0")


def gen_bin_weights(distance_df: pd.DataFrame, rel_df: pd.DataFrame, sources: np.ndarray, imdb_ffps: List[Path], n_rrup_bins: int = 10,
                    n_mag_bins: int = 10):
    """Computes samples weights based on rrup and magnitude distribution of the samples using
    a n_rrup_bins x n_mag_bins grid
    """
    mag_bins = np.linspace(rel_df.mag.min(), rel_df.mag.max(), n_mag_bins)
    rrup_bins = np.linspace(BIN_RRUP_MIN, BIN_RRUP_MAX, n_rrup_bins)

    bin_count = None
    print("Generating bin weights")
    for ix, cur_source in enumerate(sources):
        print(f"Processing {ix + 1}/{sources.size}")
        cur_im_df = nn_gmm.load_fault_im_df(cur_source, imdb_ffps) 

        if cur_im_df is None:
            print(f"WARNING: No IM data found for source {cur_source}, skipping.")
            continue

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

        # Merge with distance parameters
        cur_input_df = pd.merge(
            cur_input_df,
            distance_df,
            how="inner",
            left_on=["source", "site"],
            right_on=["source", "site"],
        )
        cur_input_df.set_index("id", inplace=True)

        cur_bin_count, _, __ = np.histogram2d(cur_input_df.rrup.values, cur_input_df.mag.values, bins=(rrup_bins, mag_bins))

        bin_count = cur_bin_count if bin_count is None else bin_count + cur_bin_count

    bin_weights = np.zeros(bin_count.shape, dtype=float)
    mask = bin_count != 0
    bin_weights[mask] = 1 / np.count_nonzero(mask) / bin_count[mask]

    assert np.isclose(np.sum(bin_weights * bin_count), 1.0)

    return bin_weights, mag_bins, rrup_bins

def main(
    site_params_dir: Path,
    distance_dir: Path,
    site_source_dir: Path,
    source_params_dir: Path,
    im_db_dir: Path,
    output_dir: str,
    val_events_ffp: Path = None,
    n_procs: int = 8,
):
    output_dir = Path(output_dir)

    # Load Site params
    print("Loading site params")
    site_df = nn_gmm.load_dfs(list(site_params_dir.glob("*.csv")))

    # Drop duplicates & check for duplicates
    n_unique_stations = np.unique(site_df.station.values.astype(str)).shape[0]
    site_df = site_df.drop_duplicates()
    assert n_unique_stations == site_df.shape[0]
    site_df.set_index("station", inplace=True)

    # Load site-source params
    print("Loading distance params")
    distance_db_ffps = list(distance_dir.glob("*.db"))
    distance_df = nn_gmm.load_distance_df(site_df, distance_db_ffps, n_procs=n_procs)

    print("Loading site-source params")
    site_source_db_ffps = list(site_source_dir.glob("*.db"))
    site_source_df = nn_gmm.load_site_source_df(site_source_db_ffps, n_procs=n_procs)
    assert (
        np.unique(site_source_df.index.values.astype(str)).shape[0]
        == site_source_df.shape[0]
    )

    # Load source params
    print("Loading realisation params")
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

        bin_weights, mag_bins, rrup_bins = gen_bin_weights(distance_df, rel_df, train_sources, im_db_ffps)

        # Training dataset
        gen_tf_records(
            train_sources,
            rel_df,
            site_df,
            distance_df,
            site_source_df,
            im_db_ffps,
            output_dir / "train",
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
            rrup_mag_weighting_data=(bin_weights, mag_bins, rrup_bins),
            n_procs=n_procs,
        )

        # Validation dataset
        gen_tf_records(
            val_sources,
            rel_df,
            site_df,
            distance_df,
            site_source_df,
            im_db_ffps,
            output_dir / "val",
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
            n_procs=n_procs,
        )
    else:
        bin_weights, mag_bins, rrup_bins = gen_bin_weights(distance_df, rel_df, rel_df.index.values.astype(str), im_db_ffps)

        gen_tf_records(
            rel_df.index.values.astype(str),
            rel_df,
            site_df,
            distance_df,
            site_source_df,
            im_db_ffps,
            output_dir,
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
            rrup_mag_weighting_data=(bin_weights, mag_bins, rrup_bins),
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
        "source_params_dir", type=str, help="The path to the source params dir",
    )
    parser.add_argument("im_db_dir", type=str, help="The path to the IM labels dbs dir")
    parser.add_argument("output_dir", type=str, help="Path of the output directory")
    parser.add_argument(
        "--val_events_ffp",
        type=str,
        help="Path to a text file that is a list of validation events (one per line)",
    )
    parser.add_argument(
        "--n_procs", type=int, help="Number of processes to use", default=4
    )
    args = parser.parse_args()

    main(
        nn_gmm.to_path(args.site_params_dir),
        nn_gmm.to_path(args.distance_dir),
        nn_gmm.to_path(args.site_source_dir),
        nn_gmm.to_path(args.source_params_dir),
        nn_gmm.to_path(args.im_db_dir),
        nn_gmm.to_path(args.output_dir),
        n_procs=args.n_procs,
        val_events_ffp=nn_gmm.to_path(args.val_events_ffp),
    )
