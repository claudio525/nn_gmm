import json
import pickle
from pathlib import Path
from typing import Union, Dict, List, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from . import data
from . import data_processing

TECT_TYPE_ONE_HOT_DICT = {"ACTIVE_SHALLOW": "active_shallow", "VOLCANIC": "volcanic"}


class GMM:
    def __init__(self, model: keras.Model, input_config: Dict):
        self.model = model
        self.input_config = input_config
        self.feature_config = input_config["feature_config"]
        self.im_config = input_config["im_config"]

        self.feature_config_prcd = data_processing.convert_to_transform_fn(
            self.feature_config.copy(), tf_fn=False
        )

        # Don't currently support pre-processing of outputs (IMs)
        assert np.all([val is None for val in self.im_config.values()])

        self.features = np.asarray(list(self.feature_config.keys()))
        self.outputs = np.asarray(list(self.im_config.keys()))

    def predict(
        self,
        X: pd.DataFrame,
        pre_process: bool = True,
        result_df_index: np.ndarray = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = self._pre_process(X) if pre_process else X

        # Ensure that all the required features exist
        if not np.all(np.isin(self.features, X.columns.values.astype(str))):
            raise ValueError("Not all required features exist in the given dataframe")

        # Run estimation
        y_est = self.model.predict(X.loc[:, self.features].values.astype(float))

        # Convert to dataframes
        result_df_index = (
            result_df_index if result_df_index is not None else X.index.values
        )
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

    def predict_dirs(
        self,
        data_dirs: List[Path],
        batch_size: int = 500_000,
        ims: List[str] = None,
        features: List[str] = []
    ):
        """
        Performs prediction using the tfrecord files in the specified directories

        Should mainly be used when getting predictions for the training or validation dataset

        Parameters
        ----------
        data_dirs: list of Path
            Directories from which to read the tfrecord files
        batch_size: int, optional
            How many records to predict in a single batch, larger will be
            faster, however requires more memory
        ims: list of strings
        features: list of strings
            The IMs and features to keep. Unless the computer used has a large
            amount of memory, keeping all is probably not the greates idea

        Returns
        -------
        sim_df: dataframe
            The simulation IM values and the features (specified in features argument)

        """
        # Get feature details, have to be same across all directories anyways
        with (data_dirs[0] / "feature_details.pickle").open("rb") as f:
            feature_details = pickle.load(f)

        ds = data.load_dataset(
            data_dirs, feature_details, batch_size=batch_size, shuffle_buffer=None
        ).prefetch(tf.data.experimental.AUTOTUNE)

        sim_dfs, mean_dfs, std_dfs = [], [], []
        for cur_data in ds.as_numpy_iterator():
            cur_df = pd.DataFrame.from_dict(cur_data)
            cur_df.set_index(cur_df.id.str.decode("UTF-8"), inplace=True)
            cur_mean_df, cur_std_df = self.predict(cur_df.copy(), pre_process=True)

            # Only keep some IMs (to reduce size of resulting data)
            if ims is not None:
                cur_mean_df, cur_std_df = cur_mean_df[ims], cur_std_df[ims]
                sim_dfs.append(cur_df[ims + features])

            mean_dfs.append(cur_mean_df)
            std_dfs.append(cur_std_df)

        mean_df, std_df = pd.concat(mean_dfs), pd.concat(std_dfs)
        return pd.concat(sim_dfs), mean_df, std_df

    def _pre_process(self, X: pd.DataFrame):
        # Deal with the categorial features
        if "tect_type" in X.columns:
            X = data_processing.apply_one_hot_enc(
                X, "tect_type", TECT_TYPE_ONE_HOT_DICT
            )

        # All other pre-processing
        X = data_processing.preprocess_df(X, self.feature_config_prcd)

        return X

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
