import pickle
import json
import os
import datetime
from typing import List, Dict, Tuple, Callable, Union
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from . import utils
from . import data_processing
from . import data
from . import model


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


def load_datasets(
    train_dirs: Union[Path, List[Path]],
    batch_size: int,
    val_dirs: Union[Path, List[Path]] = None,
    shuffle_buffer_size: int = None,
):
    """Loads the training and validation (if specified) datasets
    from the .tfrecord files in the given directories"""
    train_dirs = train_dirs if isinstance(train_dirs, list) else [train_dirs]
    with (train_dirs[0] / "feature_details.pickle").open("rb") as f:
        feature_details = pickle.load(f)

    train_ds = data.load_dataset(
        train_dirs, feature_details, batch_size, shuffle_buffer=shuffle_buffer_size
    )
    val_ds = (
        data.load_dataset(
            val_dirs, feature_details, batch_size, shuffle_buffer=shuffle_buffer_size
        )
        if val_dirs is not None
        else None
    )

    return train_ds, val_ds


def _create_output_dir(input_config: Dict):
    run_id = (
        input_config["run_id"] if "run_id" in input_config.keys() else create_run_id()
    )
    output_dir = (
        Path(input_config["output_dir"])
        if input_config["output_dir"] is not None
        else Path(input_config["base_output_dir"]) / run_id
    )
    if output_dir.is_dir():
        print(f"Ouput dir {output_dir} already exists, quitting!")
    output_dir.mkdir()

    return output_dir


def _save_configs(
    input_config: Dict, model_config: Dict, training_config: Dict, output_dir: Path
):
    with open(os.path.join(output_dir, "input_config.json"), "w") as f:
        json.dump(input_config, f, cls=utils.GenericObjJSONEncoder, indent=4)

    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, cls=utils.GenericObjJSONEncoder, indent=4)

    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(training_config, f, cls=utils.GenericObjJSONEncoder, indent=4)


def _save_model_data(
    output_dir: Path, model_dir: Path, input_config: Dict, config: Dict, history: object
):
    # Save the input (for the model)
    with open(model_dir / "input_config.json", "w") as f:
        json.dump(input_config, f, indent=4)

    # Save the model and training config
    with open(model_dir / "config.json", "w") as f:
        json.dump({key: str(value) for key, value in config.items()}, f, indent=4)

    # Save the loss
    loss_df = pd.DataFrame.from_dict(history.history)
    loss_df.index = history.epoch
    loss_df.to_csv(os.path.join(output_dir, "loss.csv"))

    return loss_df


def train(
    input_config: Dict,
    config: Dict,
    model_fn: Callable = model.create_reg_model,
    callbacks: List[keras.callbacks.Callback] = None,
    verbose: int = 2,
    multi_output: bool = False,
) -> Tuple[TrainingResult, Path]:
    """
    Runs the training based on the specified configs
    Note: Only supports training of a "single" output node model

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
    callbacks: list of keras callbacks
        Callbacks to include during training
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
    output_dir = _create_output_dir(input_config)

    # Save the input, model & training config
    _save_configs(input_config, model_config, training_config, output_dir)

    train_ds, val_ds = load_datasets(
        utils.to_path(input_config["train_data_dirs"]),
        training_config["batch_size"],
        val_dirs=utils.to_path(input_config["val_data_dirs"]),
        shuffle_buffer_size=training_config["shuffle_buffer_size"],
    )

    feature_config = data_processing.convert_to_transform_fn(
        input_config["feature_config"].copy()
    )
    im_config = data_processing.convert_to_transform_fn(
        input_config["im_config"].copy()
    )
    n_features, n_outputs = len(feature_config.keys()), len(im_config.keys())
    train_ds = data_processing.preprocess_ds(
        train_ds,
        feature_config,
        im_config,
        use_sample_weights=training_config["use_sample_weights"],
        as_dict=multi_output,
    )
    val_ds = (
        val_ds
        if val_ds is None
        else data_processing.preprocess_ds(
            val_ds, feature_config, im_config, as_dict=multi_output
        )
    )

    # Create the model
    print(f"Creating model")
    model = (
        model_fn(model_config, n_features, list(im_config.keys()))
        if multi_output
        else model_fn(model_config, n_features, n_outputs)
    )
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
    callbacks = [] if callbacks is None else callbacks
    callbacks += [
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

    # Save model data
    loss_df = _save_model_data(output_dir, model_dir, input_config, config, history)

    # Create loss plot
    plt.figure()
    plt.plot(history.epoch, loss_df.loss, label=f"Loss - {loss_df.loss.min():.4f}")
    plt.plot(
        history.epoch,
        loss_df.val_loss,
        label=f"Validation loss - {loss_df.val_loss.min():.4f}",
    )

    plt.ylabel(training_config["loss"])
    plt.xlabel("Epoch")

    plt.legend()

    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    return (
        TrainingResult(input_config, training_config, output_dir, model_dir),
        Path(output_dir),
    )
