import os
import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from . import hidden_layers

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    """
    # Load the data
    with pd.HDFStore(sample_db_ffp, mode="r") as store:
        X = store["X"]
        y = store["y"]

    # Drop the ignored features columns
    if ignore_features is not None:
        X = X.drop(columns=ignore_features)

    # One hot encoding of categorial features
    if categorial_features is not None:
        if np.isin(categorial_features, X.columns):
            X = pd.get_dummies(X, columns=categorial_features)

    return X, y


def create_model(model_config: Dict, n_inputs: int, n_outputs: int) -> keras.Model:
    """
    Creates a functional keras model from the model config

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
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        std_scaler: preprocessing.StandardScaler,
        min_max_scaler: preprocessing.MinMaxScaler,
        best_model_ffp: str,
    ):
        self.input_config = input_config
        self.training_config = training_config
        self.output_dir = output_dir

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.std_scaler = std_scaler
        self.min_max_scaler = min_max_scaler

        self.best_model_ffp = best_model_ffp


def run(input_config: Dict, train_config: Dict) -> TrainingResult:
    """
    Runs the training based on the specified configs

    Parameters
    ----------
    input_config: dictionary
        For an example see EXAMPLE_INPUT_CONFIG
    train_config: dictionary
        For an example see EXAMPLE_TRAIN_CONFIG

    Returns
    -------
    TrainingResult
    """
    print(f"================================ Training =================================")

    # Load configs
    model_config = train_config["model_config"]
    training_config = train_config["training_config"]
    batch_size, n_epochs = training_config["batch_size"], training_config["n_epochs"]

    # Create the output directory
    output_dir = input_config["output_dir"]
    if output_dir is None:
        output_dir = os.path.join(input_config["base_output_dir"], create_run_id())
    if os.path.isdir(output_dir):
        print(f"Ouput dir {output_dir} already exists, quitting!")
    os.mkdir(output_dir)

    # Load & clean the data
    print(f"Loading samples")
    X, y = load_clean_samples(
        input_config["sample_db_ffp"],
        input_config["ignore_features"],
        input_config["categorial_features"],
    )
    # X, y = X.iloc[:10000, :], y.iloc[:10000, :]

    # Split into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=training_config["val_size"]
    )
    n_train, n_val = X_train.shape[0], X_val.shape[0]
    n_features, n_outputs = X_train.shape[1], y_train.shape[1]

    # Preprocessing
    # TODO: Hmh
    # Pretty sure this error message does not apply here
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    with pd.option_context("mode.chained_assignment", None):
        std_scale_features = input_config["std_scale_features"]
        std_scaler = preprocessing.StandardScaler()
        X_train.loc[:, std_scale_features] = std_scaler.fit_transform(
            X_train.loc[:, std_scale_features].values
        )
        X_val.loc[:, std_scale_features] = std_scaler.transform(
            X_val.loc[:, std_scale_features].values
        )

        min_max_features = input_config["min_max_scale_features"]
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train.loc[:, min_max_features] = min_max_scaler.fit_transform(
            X_train.loc[:, min_max_features].values
        )
        X_val.loc[:, min_max_features] = min_max_scaler.transform(
            X_val.loc[:, min_max_features].values
        )

    # Create the train & validation datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    train_dataset = train_dataset.shuffle(n_train).batch(batch_size).prefetch(50)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))
    val_dataset = val_dataset.batch(batch_size).prefetch(50)

    # Create the model
    print(f"Creating model")
    model = create_model(model_config, n_features, n_outputs)
    model.compile(optimizer="Adam", loss="MSE")

    # Model architecture summary
    model.summary()
    keras.utils.plot_model(model, to_file=os.path.join(output_dir, "model.png"))

    # Callbacks
    best_model_ffp = os.path.join(output_dir, "best_model.h5")
    callbacks = [
        # Saves the best model (based on the validation loss)
        keras.callbacks.ModelCheckpoint(
            best_model_ffp, monitor="val_loss", save_best_only=True
        )
    ]

    # Train
    print(f"Training...")
    history = model.fit(
        train_dataset, epochs=n_epochs, validation_data=val_dataset, callbacks=callbacks
    )

    # Create loss plot
    plt.figure()
    plt.plot(history.epoch, history.history["loss"], label="Loss")
    plt.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    return TrainingResult(
        input_config,
        training_config,
        output_dir,
        X_train,
        y_train,
        X_val,
        y_val,
        std_scaler,
        min_max_scaler,
        best_model_ffp,
    )
