import json
import pickle
from pathlib import Path
from typing import Union, Dict, List, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from . import data_processing

TECT_TYPE_ONE_HOT_DICT = {"ACTIVE_SHALLOW": "active_shallow", "VOLCANIC": "volcanic"}


class GMM:
    def __init__(self, model: keras.Model, input_config: Dict):
        self.model = model
        self.input_config = input_config

        self.features = np.asarray(list(input_config["feature_config"].keys()))
        self.outputs = np.asarray(list(input_config["im_config"].keys()))

    def predict(
        self,
        X: pd.DataFrame,
        pre_process: bool = True,
        result_df_index: np.ndarray = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_config = data_processing.convert_to_transform_fn(
            self.input_config["feature_config"].copy(), tf_fn=False
        )

        if pre_process:
            # Deal with the categorial features
            X = data_processing.apply_one_hot_enc(X, "tect_type", TECT_TYPE_ONE_HOT_DICT)

            # All other pre-processing
            X = data_processing.preprocess_df(X, feature_config)

        # Ensure that all the required features exist
        if not np.all(np.isin(self.features, X.columns.values.astype(str))):
            raise ValueError("Not all required features exist in the given dataframe")

        # Run estimation
        y_est = self.model.predict(X.loc[:, self.features].values.astype(float))

        # Conver to dataframes
        mean_df = pd.DataFrame(
            data=np.exp(y_est[:, : self.outputs.size]),
            columns=self.outputs,
            index=result_df_index,
        )
        std_df = pd.DataFrame(
            data=y_est[:, self.outputs.size :],
            columns=self.outputs,
            index=result_df_index,
        )

        return mean_df, std_df

    @classmethod
    def load(cls, model_dir: Union[str, Path]):
        model_dir = model_dir if isinstance(model_dir, Path) else Path(model_dir)

        model = keras.models.load_model(
            str(model_dir), custom_objects={"MargNLLLoss": MargNLLLoss}
        )

        with open(model_dir / "input_config.json", "r") as f:
            input_config = json.load(f)

        return cls(model, input_config)


def create_gaussian_model(
    model_config: Dict, n_inputs: int, n_outputs: int
) -> keras.Model:
    """Creates a model that estimates the mean & standard deviation
    for the given target variables

    Parameters
    ----------
    model_config: dictionary
    n_inputs: int
        Number of inputs/features
    n_outputs: int
        Number of target variables, the number of
        actual model outputs will be 2 * n_outputs, since
        the model will estimate a mean & std for each
        target variable

    Returns
    -------
    keras.Model
    """
    hidden_layer_func = model_config["hidden_layer_func"]
    hidden_layer_config = model_config["hidden_layer_config"]
    units = model_config["units"]

    input = keras.Input(n_inputs)

    x = hidden_layer_func(input, units[0], **hidden_layer_config)
    for unit in units[1:]:
        x = hidden_layer_func(x, unit, **hidden_layer_config)

    mean_output = keras.layers.Dense(units=n_outputs, activation=None, name="means")(x)
    std_output = keras.layers.Dense(units=n_outputs, activation=nnelu, name="stds")(x)

    output = keras.layers.Concatenate(name="output")([mean_output, std_output])
    return keras.Model(inputs=input, outputs=output)


def create_reg_model(model_config: Dict, n_inputs: int, n_outputs: int) -> keras.Model:
    """Creates a functional keras model from the model config,
    with a linear output layer

    Parameters
    ----------
    model_config: dictionary
        Model config,
    n_inputs
    n_outputs

    Returns
    -------
    keras.Model
    """
    hidden_layer_func = model_config["hidden_layer_func"]
    hidden_layer_config = model_config["hidden_layer_config"]
    units = model_config["units"]

    input = keras.Input(n_inputs)

    x = hidden_layer_func(input, units[0], **hidden_layer_config)
    for unit in units[1:]:
        x = hidden_layer_func(x, unit, **hidden_layer_config)

    outputs = keras.layers.Dense(units=n_outputs, activation=None)(x)

    return keras.Model(inputs=input, outputs=outputs)


def nnelu(input):
    """Non-negative elu function, i.e. ELU(z) + 1"""
    return tf.add(tf.constant(1.0, dtype=tf.float32), tf.nn.elu(input))


class MargNLLLoss(keras.losses.Loss):
    def __init__(self, n_outputs: int, **kwargs):
        super().__init__(**kwargs)
        self.n_outputs = tf.constant(n_outputs, dtype=tf.int32)

    def call(self, y_true, parameters):
        means = parameters[:, : self.n_outputs]
        stds = parameters[:, self.n_outputs :]

        gaussians = tfp.distributions.Normal(loc=means, scale=stds)
        log_likelihood = gaussians.log_prob(y_true)

        return -tf.reduce_mean(log_likelihood, axis=-1)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "n_outputs": int(self.n_outputs)}
