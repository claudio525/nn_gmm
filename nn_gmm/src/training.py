import pickle
import json
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
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
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
    """
    # Load the data
    with pd.HDFStore(sample_db_ffp, mode="r") as store:
        X = store["X"]
        y = store["y"]

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

    return X, y, cat_columns


def nnelu(input: tf):
    """Non-negative elu function, i.e. ELU(z) + 1"""
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


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
    return output


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
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        best_model_dir: str,
    ):

        self.input_config = input_config
        self.training_config = training_config
        self.output_dir = output_dir

        self.X = X
        self.y = y

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.best_model_dir = best_model_dir

    def save(self, output_ffp: str):
        with open(output_ffp, "wb") as f:
            pickle.dump(self, f)


class MargNLLLoss(keras.losses.Loss):
    def __init__(self, n_outputs: int):
        super().__init__()
        self.n_outputs = tf.constant(n_outputs, dtype=tf.int32)

    def call(self, y_true, parameters):
        means = parameters[:, self.n_outputs]
        stds = parameters[:, self.n_outputs :]

        gaussians = tfp.distributions.Normal(loc=means, scale=stds)
        log_likelihood = gaussians.log_prob(tf.transpose(y_true))

        return -tf.reduce_mean(log_likelihood, axis=-1)


def train(
    input_config: Dict,
    train_config: Dict,
    model_fn: Callable = create_reg_model,
    verbose: int = 2,
) -> Tuple[TrainingResult, str]:
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

    # Load & clean the data
    print(f"Loading samples")
    X, y, cat_columns = load_clean_samples(
        input_config["sample_db_ffp"],
        input_config["ignore_features"],
        input_config["categorial_features"],
    )
    X, y = X.iloc[:10000, :], y.iloc[:10000, :]

    # Split into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=training_config["val_size"]
    )
    n_train, n_val = X_train.shape[0], X_val.shape[0]
    n_features, n_outputs = X_train.shape[1], y_train.shape[1]

    # Preprocessing of features
    # Pretty sure this error message does not apply here
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    with pd.option_context("mode.chained_assignment", None):
        std_scale_features = input_config["std_scale_features"]
        std_scaler = preprocessing.StandardScaler()
        if std_scale_features is not None and len(std_scale_features) > 0:
            X_train.loc[:, std_scale_features] = std_scaler.fit_transform(
                X_train.loc[:, std_scale_features].values
            )
            X_val.loc[:, std_scale_features] = std_scaler.transform(
                X_val.loc[:, std_scale_features].values
            )

        min_max_features = input_config["min_max_scale_features"]
        min_max_scaler = preprocessing.MinMaxScaler()
        if min_max_features is not None and len(min_max_features) > 0:
            X_train.loc[:, min_max_features] = min_max_scaler.fit_transform(
                X_train.loc[:, min_max_features].values
            )
            X_val.loc[:, min_max_features] = min_max_scaler.transform(
                X_val.loc[:, min_max_features].values
            )

    # Preprocessing of the outputs, log transform & standardise
    assert np.all(y_train.columns == y_val.columns)
    std_scaler_y = preprocessing.StandardScaler()
    y_train = pd.DataFrame(
        index=y_train.index,
        columns=y_train.columns,
        data=std_scaler_y.fit_transform(np.log(y_train.values)),
    )
    y_val = pd.DataFrame(
        index=y_val.index,
        columns=y_val.columns,
        data=std_scaler_y.transform(np.log(y_val.values)),
    )

    # Create the train & validation datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    train_dataset = train_dataset.shuffle(n_train).batch(batch_size).prefetch(50)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))
    val_dataset = val_dataset.batch(batch_size).prefetch(50)

    # Create the model
    print(f"Creating model")
    model = model_fn(model_config, n_features, n_outputs)
    model.compile(optimizer=train_config["optimizer"], loss=training_config["loss"])

    # Model architecture summary
    model.summary()

    # Callbacks
    model_dir = output_dir / "best_model"
    callbacks = [
        # Saves the best model (based on the validation loss)
        keras.callbacks.ModelCheckpoint(
            str(model_dir), monitor="val_loss", save_best_only=True
        )
    ]

    # Train
    print(f"Training...")
    history = model.fit(
        train_dataset,
        epochs=n_epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Save the feature & output scalers
    with open(model_dir / "preprocessing.pickle", "wb") as f:
        pickle.dump(
            {
                "std_scaler": std_scaler,
                "min_max_scaler": min_max_scaler,
                "std_scaler_y": std_scaler_y,
                "cat_columns": cat_columns,
            },
            f,
        )

    # Save the order of the features and output
    np.save(model_dir / "features.npy", X_train.columns.values.astype(str))
    np.save(model_dir / "outputs.npy", y_train.columns.values.astype(str))

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
            X,
            y,
            X_train,
            y_train,
            X_val,
            y_val,
            model_dir,
        ),
        output_dir,
    )
