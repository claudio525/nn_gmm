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


def nnelu(input):
    """Non-negative elu function, i.e. ELU(z) + 1"""
    return tf.add(tf.constant(1.0, dtype=tf.float32), tf.nn.elu(input))


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
        best_model_dir: str,
    ):

        self.input_config = input_config
        self.training_config = training_config
        self.output_dir = output_dir

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


def _load_dataset(
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
    parsed_dataset = (
        tf.data.Dataset.list_files(str(data_dir / "*.tfrecord"), shuffle=True)
        .interleave(
            lambda f: tf.data.TFRecordDataset(f),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            cycle_length=n_open_files,
            block_length=1,
            deterministic=False,
        )
        .shuffle(shuffle_buffer)
        .batch(batch_size)
        .map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    )

    return parsed_dataset


def load_datasets(train_dir: Path, batch_size: int, val_dir: Path = None, shuffle_buffer: int = None):
    """Loads the training and validation (if specified) datasets
    from the .tfrecord files in the given directories"""
    with (train_dir / "feature_details.pickle").open("rb") as f:
        feature_details = pickle.load(f)

    train_ds = _load_dataset(train_dir, feature_details, batch_size, shuffle_buffer=shuffle_buffer)
    val_ds = (
        _load_dataset(val_dir, feature_details, batch_size, shuffle_buffer=shuffle_buffer)
        if val_dir is not None
        else None
    )

    return train_ds, val_ds


def preprocess(ds: tf.data.Dataset, feature_config: Dict, im_config: Dict):
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


def get_standard_scaling_fn(mean: float, std: float):
    """Returns a tensorflow function for standardising
    data using the specified mean and standard deviation"""

    def standard_fn(tensor):
        return (tensor - mean) / std

    return tf.function(standard_fn)


def get_min_max_scaling_fn(
    data_min: float, data_max: float, target_min: float = 0.0, target_max: float = 1.0
):
    """Returns a tensorflow function that performs
    min-max scaling to the specified range

    Parameters
    ----------
    data_min: float
    data_max: float
        The min & max values from the unscaled data
    target_min
    target_max
        The target min & max values

    Returns
    -------
    tf.function
    """

    def min_max_fn(tensor):
        return ((tensor - data_min) / (data_max - data_min)) * (
            target_max - target_min
        ) + target_min

    return tf.function(min_max_fn)


def train(
    input_config: Dict,
    config: Dict,
    model_fn: Callable = create_reg_model,
    verbose: int = 2,
) -> Tuple[TrainingResult, str]:
    """
    Runs the training based on the specified configs

    Parameters
    ----------
    input_config: dictionary
        For an example see EXAMPLE_INPUT_CONFIG
    config: dictionary
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
    model_config = config["model_config"]
    training_config = config["training_config"]
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
        json.dump(config, f, cls=utils.GenericObjJSONEncoder)

    train_ds, val_ds = load_datasets(
        Path(input_config["train_data_dir"]),
        training_config["batch_size"],
        val_dir=Path(input_config["val_data_dir"])
        if input_config["val_data_dir"] is not None
        else None,
        shuffle_buffer=training_config["shuffle_buffer_size"]
    )

    feature_config, im_config = (
        input_config["feature_config"],
        input_config["im_config"],
    )
    n_features, n_outputs = len(feature_config.keys()), len(im_config.keys())
    train_ds = preprocess(train_ds, feature_config, im_config)
    val_ds = val_ds if val_ds is None else preprocess(val_ds, feature_config, im_config)

    # Create the model
    print(f"Creating model")
    model = model_fn(model_config, n_features, n_outputs)
    model.compile(
        optimizer=training_config["optimizer"],
        loss=training_config["loss"],
        run_eagerly=False,
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
        ),
        # keras.callbacks.TensorBoard(str(output_dir / "log"), profile_batch="2,10")
    ]

    print(f"Preparing datasets")
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # Train
    print(f"Training...")
    history = model.fit(
        train_ds,
        epochs=n_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Save the input (for the model)
    with open(model_dir / "input_config.pickle", "wb") as f:
        pickle.dump(input_config, f)

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
        TrainingResult(input_config, training_config, output_dir, model_dir),
        output_dir,
    )
