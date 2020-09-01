import pickle
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

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


def _serialize(ix_1, cur_input_row, ix_2, cur_im_df_row):
    assert ix_1 == ix_2

    features = {
        **{key: _float_feature(value) for key, value in cur_input_row.items()},
        **{key: _float_feature(value) for key, value in cur_im_df_row.items()},
    }
    features["id"] = _bytes_feature(str.encode(ix_1))

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def serialize(input_df: pd.DataFrame, im_df: pd.DataFrame, n_procs: int = 8):
    """Serializes training data (features & labels) into the tf.train.Example format"""
    ser_examples = []

    if input_df.shape[0] < 1000 or n_procs == 1:
        for (ix_1, cur_input_row), (ix_2, cur_im_df_row) in zip(
            input_df.iterrows(), im_df.iterrows()
        ):
            ser_examples.append(_serialize(ix_1, cur_input_row, ix_2, cur_im_df_row))
    else:
        with mp.Pool(processes=n_procs) as pool:
            ser_examples = pool.starmap(
                _serialize,
                [
                    (ix_1, cur_input_row, ix_2, cur_im_df_row)
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
    site_source_df: pd.DataFrame,
    im_db_ffps: List[Path],
    output_dir: Path,
    tect_type_one_hot_dict: Dict,
    n_procs: int = 8,
):
    """Generates tfrecord files using the tf.train.Example protocol,
    one file is generated per event
    """
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

        cur_input_df = nn_gmm.create_sample_comb(cur_im_df)
        cur_input_df["id"] = cur_input_df.index.values.astype(str)
        cur_input_df = pd.merge(
            # cur_input_df, source_df, how="inner", left_on="source", right_on="source"
            cur_input_df,
            rel_df,
            how="inner",
            left_on="realisation",
            right_index=True,
            suffixes=(None, "_rel_df"),
        )
        cur_input_df = pd.merge(
            cur_input_df, site_df, how="inner", left_on="site", right_index=True
        )

        cur_input_df = pd.merge(
            cur_input_df,
            site_source_df,
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

        assert np.all(
            cur_input_df.index.values.astype(str) == cur_im_df.index.values.astype(str)
        )
        examples = serialize(cur_input_df, cur_im_df[IMs], n_procs=n_procs)

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
                **{"id": tf.io.FixedLenFeature([], tf.string)},
            }
            with open(str(output_dir / "feature_details.pickle"), "wb") as f:
                pickle.dump(feature_description, f)

        with tf.io.TFRecordWriter(str(cur_output_ffp)) as writer:
            for example in examples:
                writer.write(example)


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
    site_df = nn_gmm.load_dfs(list(site_params_dir.glob("*.csv")), index_col="station")

    # Check there are no duplicates
    assert np.unique(site_df.index).shape[0] == site_df.shape[0]

    # Load site-source params
    print("Loading distance params")
    distance_db_ffps = list(distance_dir.glob("*.db"))
    distance_df = nn_gmm.load_distance_df(
        site_df, distance_db_ffps, n_procs=n_procs
    )

    print("Loading site-source params")
    site_source_db_ffps = list(distance_dir.glob("*.db"))
    site_source_df = nn_gmm


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

        # Training dataset
        gen_tf_records(
            train_sources,
            rel_df,
            site_df,
            distance_df,
            im_db_ffps,
            output_dir / "train",
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
            n_procs=n_procs,
        )

        # Validation dataset
        gen_tf_records(
            val_sources,
            rel_df,
            site_df,
            distance_df,
            im_db_ffps,
            output_dir / "val",
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
            n_procs=n_procs,
        )
    else:
        gen_tf_records(
            rel_df.index.values.astype(str),
            rel_df,
            site_df,
            distance_df,
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
    parser.add_argument("site_source_dir", type=str, help="The path to the site-source dir")
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
