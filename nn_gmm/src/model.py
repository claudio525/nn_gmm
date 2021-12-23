import json
import pickle
import time
from pathlib import Path
from typing import Union, Dict, List, Tuple

import ml_tools.utils
import xgboost as xgb
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import data
from . import data_processing
from .console import console

TECT_TYPE_ONE_HOT_DICT = {"ACTIVE_SHALLOW": "active_shallow", "VOLCANIC": "volcanic"}


class GMM:
    def __init__(self, input_config: Dict):
        self.input_config = input_config
        self.feature_config = input_config["feature_config"]
        self.im_config = input_config["im_config"]

        self.feature_config_prcd = data_processing.convert_to_transform_fn(
            self.feature_config.copy(), tf_fn=True
        )
        self.im_config_prcd = data_processing.convert_to_inv_transform_fn(
            self.im_config.copy(), tf_fn=True
        )

        self.features = np.asarray(list(self.feature_config.keys()))
        self.outputs = np.asarray(list(self.im_config.keys()))

    @property
    def ims(self):
        return list(self.im_config.keys())

    def predict(
        self,
        X: pd.DataFrame,
        pre_process: bool = True,
        result_df_index: np.ndarray = None,
    ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
        raise NotImplementedError()

    def predict_dirs(
        self,
        data_dirs: List[Path],
        batch_size: int = 1_000_000,
        ims: List[str] = None,
        features: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Union[pd.DataFrame, None]]:
        """
        Performs prediction using the tfrecord files in
        the specified directories

        Should mainly be used when getting predictions
        for the training or validation dataset

        Parameters
        ----------
        data_dirs: list of Path
            Directories from which to read the tfrecord files
        batch_size: int, optional
            How many records to predict in a single batch,
            larger will be faster, however requires more memory
        ims: list of strings
            The IMs to keep, defaults to all supported IMs of the model
            Note: If the number of IMs is large, then this will result in
            large memory usage
        features: list of strings
            The features to keep. Unless the computer used has a large
            amount of memory, keeping all is probably not the greatest idea
            Default is to return no features

        Returns
        -------
        sim_df: dataframe
            The simulation IM values and the features (specified in features argument)
        est_df: dataframe
            The estimated IM values
        std_df: dataframe
            The standard deviation of the estimated IM values, only valid for
            a NN that uses MCDropout
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, model_dir: Union[str, Path]):
        if (model_dir / "xgb.model").exists():
            return XGBoostGMM.load(model_dir)
        else:
            return NeuralNetworkGMM.load(model_dir)

    def _pre_process(self, X: pd.DataFrame):
        # Deal with the categorial features
        if "tect_type" in X.columns:
            X = data_processing.apply_one_hot_enc(
                X, "tect_type", TECT_TYPE_ONE_HOT_DICT
            )

        # All other pre-processing
        X = data_processing.preprocess_df(X, self.feature_config_prcd)
        return X

    def _post_process(self, mean_df: pd.DataFrame):
        """Performs inverse pre-processing for the model outputs
        Note: Currently only supported when predicting the mean"""
        for im in mean_df.columns.values.astype(str):
            if self.im_config_prcd[im] is not None:
                mean_df[im] = self.im_config_prcd[im](
                    tf.convert_to_tensor(
                        mean_df[im].values.astype(float), dtype=tf.float32
                    )
                )

        return mean_df


class XGBoostGMM(GMM):
    def __init__(self, model: xgb.Booster, input_config: Dict):
        super().__init__(input_config)

        self.model = model

    def predict(
        self,
        X: pd.DataFrame,
        pre_process: bool = True,
        result_df_index: np.ndarray = None,
    ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
        X = self._pre_process(X.copy()) if pre_process else X

        # Ensure that all the required features exist
        if not np.all(np.isin(self.features, X.columns.values.astype(str))):
            raise ValueError("Not all required features exist in the given dataframe")

        dX = xgb.DMatrix(X.loc[:, self.features])
        y_est = self.model.predict(dX)

        # Convert to dataframes
        result_df_index = (
            result_df_index if result_df_index is not None else X.index.values
        )
        mean_df = pd.DataFrame(data=y_est, columns=self.outputs, index=result_df_index,)

        # Apply inverse pre-processing for outputs if required
        mean_df = self._post_process(mean_df)

        return mean_df, None

    def predict_dirs(
        self,
        data_dirs: List[Path],
        batch_size: int = 1_000_000,
        ims: List[str] = None,
        features: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, None]:
        """See GMM base class for the full docstring"""
        if ims is not None:
            assert len(ims) == 1, "XGBoost models are only single-output"

        # Get feature details, have to be same across all directories anyways
        features = [] if features is None else features
        with (data_dirs[0] / "feature_details.pickle").open("rb") as f:
            feature_details = pickle.load(f)

        # Prepare dataset
        ds = data.load_dataset(
            data_dirs, feature_details, batch_size=batch_size, shuffle_buffer=None
        ).prefetch(tf.data.AUTOTUNE)

        # Run predictions
        ims = ims if ims is not None else list(self.outputs)
        sim_dfs, est_dfs = [], []
        data_columns = list(self.features) + ["id"] + list(ims)
        for ix, cur_data in enumerate(ds.as_numpy_iterator()):
            console.log(f"Processing batch - {ix + 1}")
            cur_df = pd.DataFrame.from_dict(
                {
                    cur_key: cur_value
                    for cur_key, cur_value in cur_data.items()
                    if cur_key in data_columns
                }
            )
            cur_df.set_index(cur_df.id.str.decode("UTF-8"), inplace=True)
            cur_est_df, _ = self.predict(cur_df, pre_process=True)

            sim_dfs.append(cur_df[ims + features])
            est_dfs.append(cur_est_df)
            del cur_df

        est_df = pd.concat(est_dfs)
        return pd.concat(sim_dfs), est_df, None

    @classmethod
    def load(cls, model_dir: Union[str, Path]):
        model_dir = model_dir if isinstance(model_dir, Path) else Path(model_dir)

        model = xgb.Booster()
        model.load_model(model_dir / "xgb.model")

        input_config = ml_tools.utils.load_json(model_dir / "input_config.json")

        return cls(model, input_config)


class NeuralNetworkGMM(GMM):
    def __init__(self, model: keras.Model, input_config: Dict):
        super().__init__(input_config)
        self.model = model

    def predict(
        self,
        X: pd.DataFrame,
        pre_process: bool = True,
        result_df_index: np.ndarray = None,
    ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
        X = self._pre_process(X.copy()) if pre_process else X

        # Ensure that all the required features exist
        if not np.all(np.isin(self.features, X.columns.values.astype(str))):
            raise ValueError("Not all required features exist in the given dataframe")

        # Run estimation
        y_est = self.model(X.loc[:, self.features].values.astype(float))
        # y_est = self.model.predict(X.loc[:, self.features].values.astype(float), batch_size=1024)

        # Multi-output model
        if isinstance(y_est, list):
            y_est = np.stack(y_est, axis=1).reshape(-1, self.outputs.size)

        # Convert to dataframes
        result_df_index = (
            result_df_index if result_df_index is not None else X.index.values
        )
        mean_df = pd.DataFrame(
            data=y_est[:, : self.outputs.size],
            columns=self.outputs,
            index=result_df_index,
        )

        # Apply inverse pre-processing for outputs if required
        mean_df = self._post_process(mean_df)

        # Convert to non-logged output
        # mean_df = mean_df.apply(np.exp)

        if y_est.shape[1] == 2 * self.outputs.size:
            std_df = pd.DataFrame(
                data=y_est[:, self.outputs.size :],
                columns=self.outputs,
                index=result_df_index,
            )
            assert (
                self.im_config.items()[0] == None
            ), "No post-processing currently supported when using NLL"
            return mean_df, std_df

        return mean_df, None

    def predict_dirs(
        self,
        data_dirs: List[Path],
        batch_size: int = 1_000_000,
        ims: List[str] = None,
        features: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Union[pd.DataFrame, None]]:
        """See GMM base class for the full docstring"""
        features = [] if features is None else features

        # Get feature details, have to be same across all directories anyways
        with (data_dirs[0] / "feature_details.pickle").open("rb") as f:
            feature_details = pickle.load(f)

        ds = data.load_dataset(
            data_dirs, feature_details, batch_size=batch_size, shuffle_buffer=None
        ).prefetch(tf.data.AUTOTUNE)

        ims = ims if ims is not None else list(self.outputs)
        sim_dfs, mean_dfs, std_dfs = [], [], []
        data_columns = list(self.features) + ["id"] + list(ims)
        for ix, cur_data in enumerate(ds.as_numpy_iterator()):
            console.log(f"Processing batch - {ix + 1}")
            cur_df = pd.DataFrame.from_dict(
                {
                    cur_key: cur_value
                    for cur_key, cur_value in cur_data.items()
                    if cur_key in data_columns
                }
            )
            cur_df.set_index(cur_df.id.str.decode("UTF-8"), inplace=True)
            cur_mean_df, cur_std_df = self.predict(cur_df, pre_process=True)

            # Only keep some IMs (to reduce size of resulting data)
            cur_mean_df = cur_mean_df[ims]
            cur_std_df = cur_std_df[ims] if cur_std_df is not None else None

            sim_dfs.append(cur_df[ims + features])

            mean_dfs.append(cur_mean_df)
            del cur_df

            if cur_std_df is not None:
                std_dfs.append(cur_std_df)

        mean_df = pd.concat(mean_dfs)
        std_df = pd.concat(std_dfs) if len(std_dfs) > 0 else None
        return pd.concat(sim_dfs), mean_df, std_df

    @classmethod
    def load(cls, model_dir: Union[str, Path]):
        model_dir = model_dir if isinstance(model_dir, Path) else Path(model_dir)
        model_dir = model_dir / "best_model"

        model = keras.models.load_model(str(model_dir))

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


def create_reg_multi_output_model(
    model_config: Dict, n_inputs: int, output_names: List[str]
):
    """Creates a functional keras model from the model config,
    with multiple linear outputs and possible sub-nets per output

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
    output_units = model_config.get("output_units")

    inputs = keras.Input(n_inputs, name="inputs")

    x = hidden_layer_func(inputs, units[0], **hidden_layer_config)
    for unit in units[1:]:
        x = hidden_layer_func(x, unit, **hidden_layer_config)

    outputs = []
    for cur_output_name in output_names:
        if output_units is not None:
            cur_x = hidden_layer_func(x, output_units[0], **hidden_layer_config)
            for cur_out_units in output_units[1:]:
                cur_x = hidden_layer_func(cur_x, cur_out_units, **hidden_layer_config)

            outputs.append(
                keras.layers.Dense(1, activation=None, name=cur_output_name)(cur_x)
            )

    return keras.Model(inputs=inputs, outputs=outputs)


def add_multi_output_head(
    core_model: keras.Model,
    core_model_input_name: str,
    model_config: Dict,
    output_names: List[str],
    hydra_input_t: Tuple[str, int] = None,
):
    hidden_layer_func = model_config["hidden_layer_func"]
    hidden_layer_config = model_config["hidden_layer_config"]
    output_units = model_config.get("output_units")

    core_input = keras.Input(core_model.input.shape[1:], name=core_model_input_name)
    x = core_model(core_input, training=False)

    if hydra_input_t is not None:
        hydra_input = keras.Input(hydra_input_t[1], name=hydra_input_t[0])
        x = keras.layers.concatenate([x, hydra_input])

        inputs = [core_input, hydra_input]
    else:
        inputs = core_input

    outputs = []
    for cur_output_name in output_names:
        if output_units is not None:
            cur_x = hidden_layer_func(x, output_units[0], **hidden_layer_config)
            for cur_out_units in output_units[1:]:
                cur_x = hidden_layer_func(cur_x, cur_out_units, **hidden_layer_config)

            outputs.append(
                keras.layers.Dense(1, activation=None, name=cur_output_name)(cur_x)
            )

    return keras.Model(inputs=inputs, outputs=outputs)


def nnelu(input):
    """Non-negative elu function, i.e. ELU(z) + 1"""
    return tf.add(tf.constant(1.0, dtype=tf.float32), tf.nn.elu(input))


# class MargNLLLoss(keras.losses.Loss):
#     def __init__(self, n_outputs: int, **kwargs):
#         super().__init__(**kwargs)
#         self.n_outputs = tf.constant(n_outputs, dtype=tf.int32)
#
#     def call(self, y_true, parameters):
#         means = parameters[:, : self.n_outputs]
#         stds = parameters[:, self.n_outputs :]
#
#         gaussians = tfp.distributions.Normal(loc=means, scale=stds)
#         log_likelihood = gaussians.log_prob(y_true)
#
#         return -tf.reduce_mean(log_likelihood, axis=-1)
#
#     def get_config(self):
#         base_config = super().get_config()
#         return {**base_config, "n_outputs": int(self.n_outputs)}
