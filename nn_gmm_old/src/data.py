import pickle
import time
from typing import Dict, Union, List, Sequence
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf


def load_dataset(
    data_dirs: Union[Path, List[Path]],
    feature_details: Dict,
    batch_size: int,
    file_filter: str = "*.tfrecord",
    shuffle_buffer: Union[int, None] = 1_000_000,
    n_open_files: int = 32,
    block_size: int = 256,
):
    """Performs the loading and parsing of a tensorflow dataset from
    the .tfrecord files in the given directory

    Parameters
    ----------
    data_dirs: Path, list of Path
        Directory that contains the .tfrecord files to use
    feature_details: dictionary
        Specifies how to parse the data,
        see https://www.tensorflow.org/api_docs/python/tf/io/parse_example?hl=en
        for more details
    batch_size: int
        Batch size, has to be done at this step as parsing of batches is
        much more efficient when using batches compared to single entries
    file_filter: str, optional
        The file filter to use when searching the
        specified directory for records
    shuffle_buffer: int, optional
        Size of the shuffle buffer (in number of entries) to use,
        defaults to 1 Million
        If set to None, no shuffling is performed, however data is still loaded
        from multiple .tfrecord files at once (due to interleave) meaning that
        the data is still kinda "shuffled"
    n_open_files: int, optional
        How many .tfrecord files to read concurrently using the interleave
        function (https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave)
    block_size: int, optional
        Determines how many records are loaded from a .tfrecord file in one
        interleave cylce
        Default value should be fine, increasing it will make load times faster,
        however if the shuffle buffer is not appropriately sized then this may
        result in badly shuffled data

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
    file_patterns = (
        [str(cur_dir / file_filter) for cur_dir in data_dirs]
        if isinstance(data_dirs, list)
        else str(data_dirs / file_filter)
    )
    ds = tf.data.Dataset.list_files(file_patterns, shuffle=True).interleave(
        lambda f: tf.data.TFRecordDataset(f),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        cycle_length=n_open_files,
        block_length=block_size,
        deterministic=False,
    )

    if shuffle_buffer is not None:
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.batch(batch_size).map(
        _parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return ds


def load_dataset_as_df(
    data_dir: Path, columns: Sequence[str], feature_details: Dict = None
):
    """Loads the specified columns from the dataset as dataframe"""
    if feature_details is None:
        with (data_dir / "feature_details.pickle").open("rb") as f:
            feature_details = pickle.load(f)

    ds = load_dataset([data_dir], feature_details, 1_000_000, shuffle_buffer=None)

    data = []
    for cur_batch in ds.as_numpy_iterator():
        cur_data = []
        for cur_col in columns:
            cur_data.append(cur_batch[cur_col][:, np.newaxis])

        data.append(np.concatenate(cur_data, axis=1))

    return pd.DataFrame(data=np.concatenate(data, axis=0), columns=columns)


def load_tfrecord(record_ffp: str, data_details: Dict, batch_size: int = 10_000):
    """
    Loads a single tfrecord file as a dataframe

    Parameters
    ----------
    record_ffp: string
        Path to the .tfrecord file
    data_details: dictionary
        Contains the data details required for loading
    batch_size: int, optional
        The batch size used for loading, this has to exceed the number samples
        in the .tfrecord file

    Returns
    -------
    dataframe:
        Data from the specified .tfrecord file
    """
    ds = tf.data.TFRecordDataset(filenames=[record_ffp])

    def _parse_fn(example_proto):
        parsed = tf.io.parse_example(example_proto, data_details)
        return parsed

    # Slight hack, just want to parse the whole record in one go,
    # not sure how to do this without batching...
    ds = ds.batch(batch_size).map(
        _parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dfs = [pd.DataFrame.from_dict(cur_data) for cur_data in ds.as_numpy_iterator()]
    df = pd.concat(dfs)

    df.id = df.id.str.decode("UTF-8")
    df = df.set_index("id")

    return df


def sel_rand_locations(
    data_dirs: List[Path],
    feature_details: Dict,
    n_locs: int,
    shuffle_buffer: int = 5_000_000,
):
    """Selects a set of random locations specified data

    Parameters
    ----------
    n_locs: int
        Number of locations to select

    Returns
    -------
    dataframe
        with the selected locations
        columns: [lon, lat]
    """
    ds = load_dataset(data_dirs, feature_details, n_locs, shuffle_buffer=shuffle_buffer)
    data_dict = next(ds.take(1).as_numpy_iterator())

    return pd.DataFrame(
        data=np.stack((data_dict["lon"], data_dict["lat"]), axis=1),
        columns=["lon", "lat"],
    )


def load_basin_stations(basin_dir: Path):
    basin_dict = {}
    for cur_ffp in basin_dir.glob("*.txt"):
        basin_dict[cur_ffp.stem] = np.loadtxt(cur_ffp, dtype=str)

    return basin_dict


def load_feature_details(data_dir: Path):
    with (data_dir / "feature_details.pickle").open("rb") as f:
        return pickle.load(f)


def get_base_grid_stations_mask(sites: np.ndarray):
    return np.char.startswith(sites.astype(str), "0")
