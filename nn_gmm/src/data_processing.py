from typing import Dict, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from .console import console


def get_standard_scaling_fn(mean: float, std: float, tf_fn: bool = True):
    """Returns a function for standardising
    data using the specified mean and standard deviation"""

    def standard_fn(data):
        return (data - mean) / std

    return (
        tf.function(standard_fn, experimental_relax_shapes=True)
        if tf_fn is True
        else standard_fn
    )


def get_standard_inv_scaling_fn(mean: float, std: float, tf_fn: bool = True):
    """Returns a function for computing the pre-standardised values"""
    mean, std = tf.constant(mean), tf.constant(std)

    def inv_standard_fn(scald_data):
        return (scald_data * std) + mean

    return (
        tf.function(
            inv_standard_fn,
            experimental_relax_shapes=True,
            input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32), ),
        )
        if tf_fn is True
        else inv_standard_fn
    )

def get_ln_standard_scaling_fn(mean: float, std: float, tf_fn: bool = True):
    def ln_standard_fn(data):
        return (tf.math.log(data) - mean) / std

    return (
        tf.function(ln_standard_fn, experimental_relax_shapes=True)
        if tf_fn is True
        else ln_standard_fn
    )

def get_min_max_scaling_fn(
    data_min: float,
    data_max: float,
    target_min: float = -1.0,
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
    target_min: float = -1.0,
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
        if name not in df.columns:
            print(f"Ignoring feature {name} as this is not in the dataframe!")
            continue

        df[name] = (
            func(tf.convert_to_tensor(df[name].values.astype(float)))
            if func is not None
            else df[name]
        )

    return df


def preprocess_ds(
    ds: tf.data.Dataset,
    feature_config: Dict[str, tf.function] = None,
    feature_config_dict: Dict[str, Dict[str, tf.function]] = None,
    im_config: Dict = None,
    use_sample_weights: bool = False,
    as_dict: bool = False,
):
    """
    Performs pre-processing on the specified tf.data.Dataset
    using the functions in the feature & IM config

    Items in those dictionaries have to be tf functions taking
    and returning a single tensor
    
    Parameters
    ----------
    ds: dataset
        The dataset to pre-process 
    feature_config: dictionary[string, tf.function], optional
        The config that specifies the pre-processing functions
        to apply to each feature, each function has to take 
        and return a single tensor
    feature_config_dict: dictionary[string, dictionary[string, tf.function]], optional
        Use this instead of the feature_config when there are multiple
        inputs dictionaries to be returned. 
        The inner dictionaries have the same requirements as given
        in feature_config parameter docstring.
        Only suitable when data is returned as dictionary (i.e. as_dict, 
        has to be true when using this parameter)
    im_config: dictionary[string, tf.function], optional
        Config that specifies the pre-processing of the target variable
    use_sample_weights: bool, optional
        If true then sample weights are also returned for each sample
        Can only be used with as_dict=False and feature_config (not with 
        feature_dict as that requires as_dict=True)
    as_dict: bool, optional
        If True then the data is returned as dictionary
        with feature_config, format: {inputs: X, output_name_1: y1, output_name_2: y2, ..}
        with feature_dict, format: {input_name_1: X1, input_name_2: X2, ..., 
            output_name_1: y1, output_name_2: y2, ...} 
            where input_name_1 = list(feature_dict.keys())[0]

    Returns
    -------
    dictionary or tuple
    """
    # Some sanity checks
    assert (
        feature_config is not None or feature_config_dict is not None
    ), "One of feature_config or feature_config_dict has to be specified"

    assert not (
        feature_config_dict is not None and feature_config is not None
    ), "Only one of feature_config_dict and feature_config can be set"

    assert feature_config_dict is None or (
        feature_config_dict is not None and as_dict is True
    ), "If feature_config_dict is given, then as_dict has to be True"

    def _apply_pre_config(item):
        # Single input
        if feature_config is not None:
            features = []
            for name, func in feature_config.items():
                # Hack
                if name == "rrup_ln":
                    features.append(func(item["rrup"]) if func is not None else item["rrup"])
                    continue

                features.append(func(item[name]) if func is not None else item[name])

            feature_dict = {"inputs": tf.stack(features, axis=1)}
        # Multi-input
        else:
            feature_dict = {}
            for cur_input_key, cur_feature_config in feature_config_dict.items():
                cur_features = []
                for cur_name, cur_func in cur_feature_config.items():
                    cur_features.append(
                        cur_func(item[cur_name])
                        if cur_func is not None
                        else item[cur_name]
                    )

                feature_dict[cur_input_key] = tf.stack(cur_features, axis=1)

        # Also return the target variable
        if im_config is not None:
            target_dict = {}
            for name, func in im_config.items():
                target_dict[name] = func(item[name]) if func is not None else item[name]

            # Single IM output
            if len(im_config) == 1:
                im = list(im_config.keys())[0]
                if use_sample_weights:
                    return (
                        feature_dict["inputs"],
                        target_dict[im],
                        item["sample_weight"],
                    )
                else:
                    return (
                        feature_dict["inputs"],
                        target_dict[im],
                    )
            # Multi-output
            else:
                if use_sample_weights:
                    return (
                        feature_dict["inputs"],
                        target_dict,
                        item["sample_weight"]
                    )
                else:
                    return (
                        feature_dict["inputs"],
                        target_dict
                    )

        # Only return the features (either as dictionary or Tensor)
        return feature_dict if as_dict else feature_dict["inputs"]

    return ds.map(
        tf.function(_apply_pre_config), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


def convert_to_transform_fn(config: Dict, tf_fn: bool = True):
    """Converts the items in the input config to callable
    (tensorflow) functions for pre-processing
    """
    for key, item in config.items():
        if item is None:
            continue
        elif item[0] == "standard":
            config[key] = get_standard_scaling_fn(item[1], item[2], tf_fn=tf_fn)
        elif item[0] == "min_max":
            config[key] = get_min_max_scaling_fn(item[1], item[2], tf_fn=tf_fn)
        elif item[0] == "ln":
            config[key] = tf.math.log
        elif item[0] == "ln_standard":
            config[key] = get_ln_standard_scaling_fn(item[1], item[2], tf_fn=tf_fn)
        else:
            raise ValueError(f"{item} is not a valid preprocessing config value")

    return config


def convert_to_inv_transform_fn(config: Dict, tf_fn: bool = True):
    """Converts the items in the input config to callable
    (tensorflow) functions for inverse pre-processing
    """
    for key, item in config.items():
        if item is None:
            continue
        elif item[0] == "standard":
            config[key] = get_standard_inv_scaling_fn(item[1], item[2], tf_fn=tf_fn)
        elif item[0] == "min_max":
            config[key] = get_inv_min_max_scaling_fn(item[1], item[2], tf_fn=tf_fn)
        elif item[0] == "ln":
            config[key] = tf.math.exp
        else:
            raise ValueError(f"{item} is not a valid preprocessing config value")

    return config


def apply_one_hot_enc(df: pd.DataFrame, col: str, enc_dict: Dict):
    for key, value in enc_dict.items():
        df[value] = np.zeros(df.shape[0], dtype=int)
        df.loc[df[col] == key, value] = 1

    return df.drop(columns=[col])


def get_XYZ_from_LL(lon: Union[float, np.ndarray], lat: Union[float, np.ndarray]):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return x, y, z
