import pickle
import json
import glob
import os
import datetime
from typing import List, Dict, Tuple, Callable
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from . import hidden_layers
from . import utils

EXAMPLE_INPUT_CONFIG = {
    "sample_db_ffp": "/Users/Clus/code/work/nn_gmm/data/sample_dbs/v18p6.h5",
    "base_output_dir": "/Users/Clus/code/work/nn_gmm/results/test",
    "output_dir": None,
    "ignore_features": ["rtvz"],
    "categorial_features": ["tect_type"],
    "std_scale_features": [
        "vs30",
        "z1p0",
        "z2p5",
        "dip",
        "rake",
        "width",
        "ztor",
        "mag",
        "rjb",
        "rrup",
        "rx",
        "ry",
    ],
    "min_max_scale_features": ["lat", "lon"],
}

# Config
EXAMPLE_TRAIN_CONFIG = {
    "model_config": {
        "hidden_layer_config": {"dropout": 0.25},
        "hidden_layer_func": hidden_layers.relu_BN_dropout,
        "units": [60, 60, 60],
    },
    "training_config": {"val_size": 0.1, "batch_size": 32, "n_epochs": 5},
}


def load_clean_samples(
    sample_db_ffp: str,
    ignore_features: List[str] = None,
    categorial_features: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    """
    Loads the data from the sample database, drops unwanted features
    and performs one-hot encoding for the categorial features

    Parameters
    ----------
    sample_db_ffp: str
        File path to the samples db
    ignore_features: list of strings
        Names of the features to ignore
    categorial_features: list of strings
        Names of the features to one-hot encode

    Returns
    -------
    X: pd.DataFrame
    y: pd.DataFrame
    cat_columns: list of strings
        Names of all the one-hot-encoded
        categorical columns
    station_lookup: dataframe
        Station location lookup
        index = station name, columns = [lat, lon]
    """
    # Load the data
    with pd.HDFStore(sample_db_ffp, mode="r") as store:
        X = store["X"]
        y = store["y"]

    station_lookup = utils.get_station_lookup(X)

    # Drop the ignored features columns
    if ignore_features is not None:
        X = X.drop(columns=ignore_features)

    # One hot encoding of categorial features
    cat_columns = None
    if categorial_features is not None:
        if np.isin(categorial_features, X.columns):
            X = pd.get_dummies(X, columns=categorial_features)

            # Get all the new one-hot encoded categorial columns\
            cat_columns = [
                cur_col
                for cur_col in X.columns.values.astype(str)
                if any(
                    [
                        cur_col.startswith(cat_feature)
                        for cat_feature in categorial_features
                    ]
                )
            ]

    X.drop(columns=["source", "site", "fault_id"], inplace=True)
    return X, y, cat_columns, station_lookup


def nnelu(input):
    """Non-negative elu function, i.e. ELU(z) + 1"""
    return tf.add(tf.constant(1.00000001, dtype=tf.float32), tf.nn.elu(input))


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

    x = tf.keras.layers.BatchNormalization()(input)
    x = hidden_layer_func(x, units[0], **hidden_layer_config)
    # x = hidden_layer_func(input, units[0], **hidden_layer_config)
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


def create_run_id() -> str:
    """Creates a run ID based on the month, day & time"""
    id = datetime.datetime.now().strftime("%m%d_%H%M")
    return id


class TrainingResult:
    def __init__(
        self,
        input_config: Dict,
        training_config: Dict,
        output_dir: str,
        # ids: np.ndarray,
        # station_lookup: pd.DataFrame,
        # ids_train: np.ndarray,
        # ids_val: np.ndarray,
        best_model_dir: str,
    ):

        self.input_config = input_config
        self.training_config = training_config
        self.output_dir = output_dir

        # self.ids = ids
        # self.station_lookup = station_lookup
        #
        # self.ids_train = ids_train
        # self.ids_val = ids_val

        self.best_model_dir = best_model_dir

    def save(self, output_ffp: str):
        with open(output_ffp, "wb") as f:
            pickle.dump(self, f)


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

def _load_dataset(data_dir: Path, feature_details: Dict):
    data_files = glob.glob(str(data_dir / "*.tfrecord"))
    raw_dataset = tf.data.TFRecordDataset(data_files)

    def _parse_fn(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_details)
        return parsed

    parsed_dataset = raw_dataset.map(_parse_fn)

    return parsed_dataset

def load_datasets(train_dir: Path, val_dir: Path = None):
    with (train_dir / "feature_details.pickle").open("rb") as f:
        feature_details = pickle.load(f)

    train_ds = _load_dataset(train_dir, feature_details)
    val_ds = _load_dataset(val_dir, feature_details) if val_dir is not None else None

    return train_ds, val_ds

def preprocess(ds: tf.data.Dataset, feature_config: Dict, im_config: Dict):
    def _apply_pre_config(item):
        features = []
        for name, func in feature_config.items():
            features.append(func(item[name]) if func is not None else item[name])

        target_values = []
        for name, func in im_config.items():
            target_values.append(func(item[name]) if func is not None else item[name])

        return tf.stack(features), tf.stack(target_values)

    return ds.map(_apply_pre_config, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def train(
    input_config: Dict,
    train_config: Dict,
    model_fn: Callable = create_reg_model,
    verbose: int = 2,
) -> Tuple[
    TrainingResult,
    str,
    # Tuple[pd.DataFrame, pd.DataFrame],
    # Tuple[pd.DataFrame, pd.DataFrame],
]:
    """
    Runs the training based on the specified configs

    Parameters
    ----------
    input_config: dictionary
        For an example see EXAMPLE_INPUT_CONFIG
    train_config: dictionary
        For an example see EXAMPLE_TRAIN_CONFIG
    model_fn: callabel, optional
        Function that returns a keras model to train,
        must take 3 inputs: model_config, n_inputs, n_outputs
        Defaults to "create_reg_model"
    verbose: int, optional
        Model fitting verbosity for details, see
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

    Returns
    -------
    TrainingResult
    """
    print(
        f"================================ Training ================================="
    )

    # Load configs
    model_config = train_config["model_config"]
    training_config = train_config["training_config"]
    batch_size, n_epochs = training_config["batch_size"], training_config["n_epochs"]

    # Create the output directory
    output_dir = (
        Path(input_config["output_dir"])
        if input_config["output_dir"] is not None
        else Path(input_config["base_output_dir"]) / create_run_id()
    )
    if output_dir.is_dir():
        print(f"Ouput dir {output_dir} already exists, quitting!")
    output_dir.mkdir()

    # Save the input, model & training config
    with open(os.path.join(output_dir, "input_config.json"), "w") as f:
        json.dump(input_config, f, cls=utils.GenericObjJSONEncoder)

    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, cls=utils.GenericObjJSONEncoder)

    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, cls=utils.GenericObjJSONEncoder)

    train_ds, val_ds = load_datasets(Path(input_config["train_data_dir"]), Path(input_config["val_data_dir"]))

    feature_config, im_config = input_config["feature_config"], input_config["im_config"]
    n_features, n_outputs = len(feature_config.keys()), len(im_config.keys())
    train_ds = preprocess(train_ds, feature_config, im_config)
    val_ds = val_ds if val_ds is None else preprocess(val_ds, feature_config, im_config)

    # Create the model
    print(f"Creating model")
    model = model_fn(model_config, n_features, n_outputs)
    # model.compile(optimizer=training_config["optimizer"], loss=training_config["loss"])
    model.compile(
        optimizer=training_config["optimizer"],
        loss=training_config["loss"],
        run_eagerly=True,
    )

    # Model architecture summary
    model.summary()

    # Callbacks
    model_dir = output_dir / "best_model"
    model_dir.mkdir()
    callbacks = [
        # Saves the best model (based on the validation loss)
        keras.callbacks.ModelCheckpoint(
            str(model_dir), monitor="val_loss", save_best_only=True
        )
    ]

    train_ds = train_ds.shuffle(int(2e4)).batch(training_config["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(training_config["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)

    # Train
    print(f"Training...")
    history = model.fit(
        train_ds,
        epochs=n_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Save the feature & output scalers
    # with open(model_dir / "preprocessing.pickle", "wb") as f:
    #     pickle.dump(
    #         {
    #             "std_scaler": std_scaler,
    #             "min_max_scaler": min_max_scaler,
    #             # "std_scaler_y": std_scaler_y,
    #             "cat_columns": cat_columns,
    #         },
    #         f,
    #     )

    # Save the order of the features and output
    # np.save(model_dir / "features.npy", X_train.columns.values.astype(str))
    # np.save(model_dir / "outputs.npy", y_train.columns.values.astype(str))

    # Save the input (for the model)
    with open(model_dir / "input_config.json", "w") as f:
        json.dump(input_config, f)

    # Save the loss
    loss_df = pd.DataFrame.from_dict(history.history)
    loss_df.to_csv(os.path.join(output_dir, "loss.csv"))

    # Create loss plot
    plt.figure()
    plt.plot(history.epoch, loss_df.loss, label="Loss")
    plt.plot(history.epoch, loss_df.val_loss, label="Validation loss")

    plt.ylabel(training_config["loss"])
    plt.xlabel("Epoch")

    plt.legend()

    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    return (
        TrainingResult(
            input_config,
            training_config,
            output_dir,
            # ids,
            # station_lookup,
            # ids_train,
            # ids_val,
            model_dir,
        ),
        output_dir,
        # (X_train, y_train),
        # (X_val, y_val),
    )
