import pickle
import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import interpolate
from sklearn.model_selection import train_test_split

import nn_gmm

IMs = np.asarray(['AI',
 'CAV',
 'Ds575',
 'Ds595',
 'MMI',
 'PGA',
 'PGV',
 'pSA_0.01',
 'pSA_0.02',
 'pSA_0.03',
 'pSA_0.04',
 'pSA_0.05',
 'pSA_0.075',
 'pSA_0.1',
 'pSA_0.12',
 'pSA_0.15',
 'pSA_0.17',
 'pSA_0.2',
 'pSA_0.25',
 'pSA_0.3',
 'pSA_0.4',
 'pSA_0.5',
 'pSA_0.6',
 'pSA_0.7',
 'pSA_0.75',
 'pSA_0.8',
 'pSA_0.9',
 'pSA_1.0',
 'pSA_1.25',
 'pSA_1.5',
 'pSA_2.0',
 'pSA_2.5',
 'pSA_3.0',
 'pSA_4.0',
 'pSA_5.0',
 'pSA_6.0',
 'pSA_7.5',
 'pSA_10.0'])


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


def serialize(input_df: pd.DataFrame, im_df: pd.DataFrame):
    """Serializes training data (features & labels) into the tf.train.Example format"""
    ser_examples = []

    for (ix_1, cur_input_row), (ix_2, cur_im_df_row) in zip(input_df.iterrows(), im_df.iterrows()):
        assert ix_1 == ix_2

        features = {**{key: _float_feature(value) for key, value in cur_input_row.items()},
                    **{key: _float_feature(value) for key, value in cur_im_df_row.items()}}
        features["id"] = _bytes_feature(str.encode(ix_1))

        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        ser_examples.append(example_proto.SerializeToString())

    return ser_examples


def interpolate_pSA_periods(im_df: pd.DataFrame, target_ims):
    """Selects the pSA periods of interest if available in the specified
    IM dataframe, otherwise interpolates to get the desired range of
    pSA periods"""
    if np.all(np.isin(target_ims, im_df.columns)):
        return im_df

    ims = im_df.columns.values.astype(str)
    pSA_mask = np.char.startswith(ims, "pSA_")
    pSA_periods = np.stack(np.char.split(ims[pSA_mask], "_"))[:, 1].astype(float)

    target_mask = np.char.startswith(target_ims, "pSA_")
    target_periods = np.sort(np.stack(np.char.split(IMs[target_mask], "_"))[:, 1].astype(float))

    # Interpolate
    assert np.all(np.sort(pSA_periods) == pSA_periods)
    f = interpolate.interp1d(np.log(pSA_periods), im_df.loc[:, ims[pSA_mask]].values,
                             kind="linear", bounds_error=True)
    target_values = f(np.log(target_periods))

    pSA_df = pd.DataFrame(columns=np.char.add("pSA_", target_periods.astype(str)), data=target_values, index=im_df.index)
    result_df = pd.merge(im_df.loc[:, ims[~pSA_mask]], pSA_df, left_index=True, right_index=True, how="inner")

    assert result_df.shape[0] == im_df.shape[0]
    return result_df

def gen_tf_records(
    sources: np.ndarray,
    source_df: pd.DataFrame,
    site_df: pd.DataFrame,
    site_source_df: pd.DataFrame,
    im_db_ffp: str,
    output_dir: Path,
    tect_type_one_hot_dict: Dict,
):
    """Generates tfrecord files using the tf.train.Example protocol,
    one file is generated per event
    """
    for ix, cur_source in enumerate(sources):
        print(f"Processing {ix + 1}/{sources.size}")
        cur_im_df = nn_gmm.load_fault_im_df(cur_source, im_db_ffp)
        if cur_im_df is None:
            print(f"No IM data found for source {cur_source}, skipping.")
            continue
        cur_im_df.sort_index(inplace=True)

        cur_input_df = nn_gmm.create_sample_comb(cur_im_df)
        cur_input_df = pd.merge(
            cur_input_df, source_df, how="inner", left_on="source", right_index=True
        )
        cur_input_df = pd.merge(
            cur_input_df, site_df, how="inner", left_on="site", right_index=True
        )
        cur_input_df["id"] = cur_input_df.index.values.astype(str)

        cur_input_df = pd.merge(
            cur_input_df,
            site_source_df,
            how="inner",
            left_on=["source", "site"],
            right_on=["source", "site"],
        )
        cur_input_df.set_index("id", inplace=True)
        cur_input_df.sort_index(inplace=True)
        cur_input_df = cur_input_df.drop(columns=["source", "site", "rtvz"])
        # cur_input_df = cur_input_df.drop(columns=["source", "site"])
        cur_input_df = nn_gmm.apply_one_hot_enc(
            cur_input_df, "tect_type", tect_type_one_hot_dict
        )

        cur_im_df = interpolate_pSA_periods(cur_im_df, IMs)

        assert np.all(
            cur_input_df.index.values.astype(str) == cur_im_df.index.values.astype(str)
        )
        examples = serialize(cur_input_df, cur_im_df[IMs])

        if ix == 0:
            print(f"Writing feature details")
            feature_description = {
                **{col: tf.io.FixedLenFeature([], tf.float32) for col in
                   cur_input_df.columns.values.astype(str)},
                **{col: tf.io.FixedLenFeature([], tf.float32) for col in
                   cur_im_df.columns.values.astype(str)}}
            feature_description = {**feature_description, **{"id": tf.io.FixedLenFeature([], tf.string)}}
            with open(str(output_dir / "feature_details.pickle"), "wb") as f:
                pickle.dump(feature_description, f)

        with tf.io.TFRecordWriter(str(output_dir / f"{cur_source}.tfrecord")) as writer:
            for example in examples:
                writer.write(example)



def main(
    site_params_ffp: str,
    site_source_ffp: str,
    source_params_ffp: str,
    im_db_ffp: str,
    output_dir: str,
    val_prop: float = 0.1,
    n_procs: int = 4,
):
    output_dir = Path(output_dir)

    # Load Site params
    print("Loading site params")
    site_df = pd.read_csv(site_params_ffp, index_col="station")

    # Load site-source params
    print("Loading site-source params")
    # site_source_df = None
    site_source_df = nn_gmm.load_site_source_df(
        site_df, site_source_ffp, n_procs=n_procs
    )

    # Load source params
    print("Loading source params")
    source_df = pd.read_csv(source_params_ffp, index_col="fault")

    # Split events/sources into train/validation data
    if val_prop is not None and val_prop > 0.0:
        train_dir, val_dir = (output_dir / "train"), (output_dir / "val")
        train_dir.mkdir()
        val_dir.mkdir()

        # Split
        train_sources, val_sources = train_test_split(
            source_df.index.values.astype(str), test_size=val_prop
        )

        gen_tf_records(
            train_sources,
            source_df,
            site_df,
            site_source_df,
            im_db_ffp,
            train_dir,
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
        )
        gen_tf_records(
            val_sources,
            source_df,
            site_df,
            site_source_df,
            im_db_ffp,
            val_dir,
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
        )
    else:
        gen_tf_records(
            source_df.index.values.astype(str),
            source_df,
            site_df,
            site_source_df,
            im_db_ffp,
            output_dir,
            nn_gmm.TECT_TYPE_ONE_HOT_DICT,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "site_params_ffp", type=str, help="The path to the site params csv file to use"
    )
    parser.add_argument(
        "site_source_ffp", type=str, help="The path to the site-source db to use"
    )
    parser.add_argument(
        "source_params_ffp",
        type=str,
        help="The path to the source params csv file to use",
    )
    parser.add_argument("im_db_ffp", type=str, help="The path to the IM labels db")
    parser.add_argument("output_dir", type=str, help="Path of the output directory")
    parser.add_argument(
        "--n_procs", type=int, help="Number of processes to use", default=4
    )

    args = parser.parse_args()

    main(
        args.site_params_ffp,
        args.site_source_ffp,
        args.source_params_ffp,
        args.im_db_ffp,
        args.output_dir,
        n_procs=args.n_procs,
    )
