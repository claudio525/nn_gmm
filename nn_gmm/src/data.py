import pickle
from typing import Dict
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf


def load_dataset(
    data_dir: Path,
    feature_details: Dict,
    batch_size: int,
    shuffle_buffer: int = 1_000_000,
    n_open_files: int = 32,
):
    """
    Performs the loading and parsing of a tensorflow dataset from
    the .tfrecord files in the given directory

    Parameters
    ----------
    data_dir: Path
        Directory that contains the .tfrecord files to use
    feature_details: dictionary
        Specifies how to parse the data,
        see https://www.tensorflow.org/api_docs/python/tf/io/parse_example?hl=en
        for more details
    batch_size: int
        Batch size, has to be done at this step as parsing of batches is
        much more efficient when using batches compared to single entries
    shuffle_buffer: int, optional
        Size of the shuffle buffer (in number of entries) to use,
        defaults to 1 Million
        If set to None, no shuffling is performed, however data is still loaded
        from multiple .tfrecord files at once (due to interleave) meaning that
        the data is still kinda "shuffled"
    n_open_files: int, optional
        How many .tfrecord files to read concurrently using the interleave
        function (https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave)

    Returns
    -------
    tf.data.Dataset
        Note: The dataset will not return single training samples,
        but instead batches of batch_size!
    """

    def _parse_fn(example_proto):
        parsed = tf.io.parse_example(example_proto, feature_details)
        return parsed

    # 1) Finds all .tfrecord files in the given directory, the shuffle option
    # means that the order of the files is shuffled every repeat (i.e. epoch) of the
    # dataset
    # 2) Uses interleave cycle through the n_open_files .tfrecord files and feed one
    # one serialized sample into the shuffle buffer, opening new the next .tfrecord file
    # once one runs out of samples
    # 3) Shuffles all samples in the buffer and adding more into the buffer
    # as samples are removed
    # 4) Batch
    # 5) Parse each batch
    ds = tf.data.Dataset.list_files(
        str(data_dir / "*.tfrecord"), shuffle=True
    ).interleave(
        lambda f: tf.data.TFRecordDataset(f),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        cycle_length=n_open_files,
        block_length=1,
        deterministic=False,
    )

    if shuffle_buffer is not None:
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.batch(batch_size).map(
        _parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return ds


def sel_rand_locations(
    data_dir: Path, feature_details: Dict, n_locs: int, shuffle_buffer: int = 5_000_000
):
    """Selects a set of random locations from the
    specified dataframe

    Parameters
    ----------
    ds: tensorflow dataset
        Tensorflow dataset from which to select locations,
        assumes that samples are already shuffled
    n_locs: int
        Number of locations to select

    Returns
    -------
    dataframe
        with the selected locations
        columns: [lon, lat]
    """
    ds = load_dataset(data_dir, feature_details, n_locs, shuffle_buffer=shuffle_buffer)
    data_dict = next(ds.take(1).as_numpy_iterator())

    return pd.DataFrame(
        data=np.stack((data_dict["lon"], data_dict["lat"]), axis=1),
        columns=["lon", "lat"],
    )


def load_feature_details(train_dir: Path):
    with (train_dir / "feature_details.pickle").open("rb") as f:
        return pickle.load(f)
