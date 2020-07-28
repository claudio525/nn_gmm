from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf


def get_standard_scaling_fn(mean: float, std: float, tf_fn: bool = True):
    """Returns a function for standardising
    data using the specified mean and standard deviation"""

    def standard_fn(data):
        return (data - mean) / std

    return tf.function(standard_fn) if tf_fn is True else standard_fn


def get_standard_inv_scaling_fn(mean: float, std: float, tf_fn: bool = True):
    """Returns a function for computing the pre-standardised values"""

    def inv_standard_fn(scald_data):
        return (scald_data * std) + mean

    return tf.function(inv_standard_fn) if tf_fn is True else inv_standard_fn


def get_min_max_scaling_fn(
    data_min: float,
    data_max: float,
    target_min: float = 0.0,
    target_max: float = 1.0,
    tf_fn: bool = True,
):
    """Returns a function that performs
    min-max scaling to the specified range

    Parameters
    ----------
    data_min: float
    data_max: float
        The min & max values from the unscaled data
    target_min: float, optional
    target_max: float, optional
        The target min & max values
    tf_fn: bool, optional
        If true a tf.function is returned, otherwise
        a standard python function is returned

    Returns
    -------
    tf.function or function
    """
    def min_max_fn(data):
        return ((data - data_min) / (data_max - data_min)) * (
            target_max - target_min
        ) + target_min

    return tf.function(min_max_fn) if tf_fn is True else min_max_fn


def get_inv_min_max_scaling_fn(
    data_min: float,
    data_max: float,
    target_min: float = 0.0,
    target_max: float = 1.0,
    tf_fn: bool = True,
):
    """Returns a function that computes pre-min-max scaled values"""

    def inv_min_max_fn(scaled_data):
        return (scaled_data - target_min) / (target_max - target_min) * (
            data_max - data_min
        ) + data_min

    return tf.function(inv_min_max_fn) if tf_fn is True else inv_min_max_fn


def preprocess_df(df: pd.DataFrame, config: Dict):
    for name, func in config.items():
        df[name] = func(df[name].values) if func is not None else df[name]

    return df


def preprocess_ds(ds: tf.data.Dataset, feature_config: Dict, im_config: Dict):
    """Performs pre-processing on the specified tf.data.Dataset
    using the functions in the feature & IM config

    Items in those dictionaries have to be tf functions taking
    and returning a single tensor
    """

    def _apply_pre_config(item):
        features = []
        for name, func in feature_config.items():
            features.append(func(item[name]) if func is not None else item[name])

        target_values = []
        for name, func in im_config.items():
            target_values.append(func(item[name]) if func is not None else item[name])

        return tf.stack(features, axis=1), tf.stack(target_values, axis=1)

    return ds.map(
        tf.function(_apply_pre_config), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


def convert_to_transform_fn(config: Dict, tf_fn: bool = True):
    """Converts the items in the input config to callable
    tensorflow functions for pre-processing

    Note: This function is only for the training workflow (i.e. when
    using tf.data.Dataset, NOT for predictions
    """
    for key, item in config.items():
        if item is None:
            continue
        elif item[0] == "standard":
            config[key] = get_standard_scaling_fn(item[1], item[2], tf_fn=tf_fn)
        elif item[0] == "min_max":
            config[key] = get_min_max_scaling_fn(item[1], item[2], tf_fn=tf_fn)
        else:
            raise ValueError(f"{item} is not a valid preprocessing config value")

    return config


def apply_one_hot_enc(df: pd.DataFrame, col: str, enc_dict: Dict):
    for key, value in enc_dict.items():
        df[value] = np.zeros(df.shape[0], dtype=int)
        df.loc[df[col] == key, value] = 1

    return df.drop(columns=[col])
