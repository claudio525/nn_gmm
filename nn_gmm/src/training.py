import time
import pickle
import json
import os
import datetime
from typing import List, Dict, Tuple, Callable, Union, Sequence
from pathlib import Path

import numpy as np
import ml_tools.utils
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import ml_tools as mlt

from . import utils
from . import data_processing
from . import data
from . import eval
from .console import console


def create_run_id(tags: Sequence[str]) -> str:
    """Creates a run ID based on the month, day & time"""
    id = datetime.datetime.now().strftime("%m%d_%H%M%S")

    if tags is not None and len(tags) > 0:
        id = id + "_" + "_".join(tags)

    return id


class TrainingResult:
    def __init__(
        self,
        input_config: Dict,
        training_config: Dict,
        output_dir: Path,
        best_model_dir: Path,
        loss_df: pd.DataFrame
    ):

        self.input_config = input_config
        self.training_config = training_config
        self.output_dir = output_dir

        self.model_location = best_model_dir

        self.loss_df = loss_df

    def save(self, output_ffp: str):
        with open(output_ffp, "wb") as f:
            pickle.dump(self, f)


def load_datasets(
    train_dirs: Union[Path, List[Path]],
    batch_size: int,
    val_dirs: Union[Path, List[Path]] = None,
    shuffle_buffer_size: int = None,
    n_open_files: int = 512,
):
    """Loads the training and validation (if specified) datasets
    from the .tfrecord files in the given directories"""
    train_dirs = train_dirs if isinstance(train_dirs, list) else [train_dirs]
    with (train_dirs[0] / "feature_details.pickle").open("rb") as f:
        feature_details = pickle.load(f)

    train_ds = data.load_dataset(
        train_dirs,
        feature_details,
        batch_size,
        shuffle_buffer=shuffle_buffer_size,
        n_open_files=n_open_files,
    )
    val_ds = (
        data.load_dataset(
            val_dirs,
            feature_details,
            batch_size,
            shuffle_buffer=shuffle_buffer_size,
            n_open_files=n_open_files,
        )
        if val_dirs is not None
        else None
    )

    return train_ds, val_ds


def create_output_dir(config: Dict) -> Path:
    run_id = config["run_id"] if "run_id" in config.keys() else create_run_id()
    output_dir = (
        Path(config["output_dir"])
        if config.get("output_dir") is not None
        else Path(config["base_output_dir"]) / run_id
    )
    if output_dir.is_dir():
        print(f"Ouput dir {output_dir} already exists, quitting!")
    output_dir.mkdir()

    return Path(output_dir)


def _save_configs(
    io_config: Dict, model_config: Dict, training_config: Dict, output_dir: Path
):
    ml_tools.utils.write_to_json(io_config, output_dir / "input_config.json")

    if model_config is not None:
        ml_tools.utils.write_to_json(model_config, output_dir / "model_config.json")

    ml_tools.utils.write_to_json(training_config, output_dir / "train_config.json")


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


def train_nn(
    config: Dict,
    hyperparams: Dict,
    model: keras.Model = None,
    model_fn: Callable = None,
    model_config: Dict = None,
    verbose: int = 2,
    as_dict: bool = False,
    output_dir: Path = None,
) -> TrainingResult:
    """
    Runs the training based on the specified configs
    Note: Only supports training of a "single" output node model

    Parameters
    ----------
    data_config: dictonary
        The input and output data config
    config: dictionary
        The model inputs and outputs config
    hyperparams: dictionary
        The training config
    model: keras model, optional
        The model to train
        Either the model or model_specs parameter has to be specified
    model_fn: callabel, optional
        The callable must be a function that returns the keras model to train
        and takes 3 inputs: model_config, n_inputs, n_outputs
        Either the model or model_specs parameter has to be specified
    model_config: dictionary
        The dictionary specifies the details of the model, as required
        by the model creation function
        If a model is passed then this config is not used, only saved
        with the model
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

    assert (
        model is not None or model_fn is not None
    ), "Either the model or model_specs parameter has to be specified"

    assert (model_fn is None) or (
        model_fn is not None and model_config is not None
    ), "If a model creation function is used, then a model config has to be specified"

    # Load hyperparamters
    batch_size, n_epochs = hyperparams["batch_size"], hyperparams["n_epochs"]

    # Create the output directory
    output_dir = create_output_dir(config) if output_dir is None else output_dir

    # Save the input, model & training config
    _save_configs(config, model_config, hyperparams, output_dir)

    train_ds, val_ds = load_datasets(
        utils.to_path(config["train_data_dirs"]),
        hyperparams["batch_size"],
        val_dirs=utils.to_path(config["val_data_dirs"]),
        shuffle_buffer_size=hyperparams["shuffle_buffer_size"],
        # shuffle_buffer_size=None,
        n_open_files=512,
    )

    # Pre-processing
    if config.get("multi_input") is not None:
        feature_config_dict, feature_config = (
            {
                cur_input_name: data_processing.convert_to_transform_fn(
                    cur_feature_config.copy()
                )
                for cur_input_name, cur_feature_config in config[
                    "feature_config"
                ].items()
            },
            None,
        )
    else:
        feature_config_dict, feature_config = (
            None,
            data_processing.convert_to_transform_fn(config["feature_config"].copy()),
        )
    im_config = data_processing.convert_to_transform_fn(config["im_config"].copy())


    train_ds = data_processing.preprocess_ds(
        train_ds,
        feature_config=feature_config,
        feature_config_dict=feature_config_dict,
        im_config=im_config,
        use_sample_weights=config["use_sample_weights"],
        as_dict=as_dict,
    )
    val_ds = (
        val_ds
        if val_ds is None
        else data_processing.preprocess_ds(
            val_ds,
            feature_config=feature_config,
            feature_config_dict=feature_config_dict,
            im_config=im_config,
            as_dict=as_dict,
        )
    )

    # Cache if specified
    if hyperparams["cache"]:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    # Shuffle the batches
    train_ds.shuffle(128)

    console.print(f"Preparing datasets")
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # Compile the model
    model.compile(
        optimizer=hyperparams["optimizer"],
        loss=hyperparams["loss"],
        run_eagerly=True,
    )

    # Model architecture summary
    model.summary()

    # Save a plot of the model
    keras.utils.plot_model(
        model,
        output_dir / "model.png",
        expand_nested=True,
        show_dtype=True,
        show_shapes=True,
    )

    # Callbacks
    model_dir = output_dir / "best_model"
    model_dir.mkdir()
    callbacks = [] if hyperparams.get("callbacks") is None else hyperparams["callbacks"]
    if hyperparams.get("save_best_val") is True:
        callbacks += [
            # Saves the best model (based on the validation loss)
            keras.callbacks.ModelCheckpoint(
                str(model_dir), monitor="val_loss", save_best_only=True
            ),
        ]

    # Train
    console.print(f"Training...")
    history = model.fit(
        train_ds,
        epochs=n_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=verbose,
    )

    if not hyperparams.get("save_best_val"):
        print("Saving the model")
        model.save(model_dir, save_format="tf")

    # Save model data
    loss_df = _save_model_data(output_dir, model_dir, config, hyperparams, history)

    # Create loss plot
    history = history.history
    ims = list(im_config.keys())

    if config["use_sample_weights"]:
        fig = plt.figure(figsize=(16, 10), dpi=200)
        ax_1 = fig.add_subplot(1, 2, 1)
        ml_tools.plotting.plot_loss(
            history,
            ax=ax_1,
            y_label="Training Loss",
            multi_keys=ims if len(ims) > 1 else None,
            plot_val=False
        )

        ax_2 = fig.add_subplot(1, 2, 2)
        ml_tools.plotting.plot_loss(
            history,
            ax=ax_2,
            y_label="Validation Loss",
            multi_keys=ims if len(ims) > 1 else None,
            plot_train=False,
        )
    else:
        fig = ml_tools.plotting.plot_loss(
            history,
            y_label="Loss",
            multi_keys=ims if len(ims) > 1 else None,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss.png"))

    return TrainingResult(config, hyperparams, output_dir, model_dir, loss_df)


def get_loss_function(hyperparams: Dict):
    loss = hyperparams["loss_key"].lower()
    if loss == "mse":
        return tf.losses.MeanSquaredError()
    if loss == "huber":
        return tf.losses.Huber(delta=hyperparams["delta"])


# class XGBDataIterator(xgb.DataIter):
#     def __init__(self, tf_ds: tf.data.Dataset):
#         super().__init__()
#
#         self.ds = tf_ds
#         self.cur_gen = self.batch_iterator()
#
#     def batch_iterator(self):
#         for cur_batch in self.ds.as_numpy_iterator():
#             yield cur_batch
#
#     def next(self, input_data: Callable) -> int:
#         try:
#             X, y = next(self.cur_gen)
#         except StopIteration:
#             return 0
#
#         input_data(data=X, label=y.ravel())
#         return 1
#
#     def reset(self) -> None:
#         self.cur_gen.close()
#
#         self.cur_gen = self.batch_iterator()
#
#
# def train_xgb(
#     io_config: Dict, train_config: Dict, model_params: Dict, output_dir: Path = None,
# ):
#     def xgb_mse_metric(pred: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
#         y = dtrain.get_label()
#         return "MSE", eval.mse(y, pred)
#
#     # Load hyperparamters
#     # batch_size, n_epochs = training_config["batch_size"], training_config["n_epochs"]
#
#     # Create the output directory
#     output_dir = create_output_dir(io_config) if output_dir is None else output_dir
#
#     # Save the input, model & training config
#     _save_configs(io_config, model_params, train_config, output_dir)
#
#     train_ds, val_ds = load_datasets(
#         utils.to_path(io_config["train_data_dirs"]),
#         train_config["batch_size"],
#         val_dirs=utils.to_path(io_config["val_data_dirs"]),
#         shuffle_buffer_size=train_config["shuffle_buffer_size"],
#         n_open_files=512,
#     )
#
#     # Pre-processing
#     console.print("Preprocessing")
#     if io_config.get("multi_input") is not None:
#         feature_config_dict, feature_config = (
#             {
#                 cur_input_name: data_processing.convert_to_transform_fn(
#                     cur_feature_config.copy()
#                 )
#                 for cur_input_name, cur_feature_config in io_config[
#                     "feature_config"
#                 ].items()
#             },
#             None,
#         )
#     else:
#         feature_config_dict, feature_config = (
#             None,
#             data_processing.convert_to_transform_fn(io_config["feature_config"].copy()),
#         )
#     im_config = data_processing.convert_to_transform_fn(io_config["im_config"].copy())
#     train_ds = data_processing.preprocess_ds(
#         train_ds,
#         feature_config=feature_config,
#         feature_config_dict=feature_config_dict,
#         im_config=im_config,
#         use_sample_weights=train_config["use_sample_weights"],
#     )
#     val_ds = (
#         val_ds
#         if val_ds is None
#         else data_processing.preprocess_ds(
#             val_ds,
#             feature_config=feature_config,
#             feature_config_dict=feature_config_dict,
#             im_config=im_config,
#         )
#     )
#
#     # Preparing dataset for XGBoost
#     console.print(f"Preparing datasets")
#     train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
#     val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)
#
#     if train_config["cache"]:
#         train_ds = train_ds.cache()
#         val_ds = val_ds.cache()
#
#     console.print("Loading data for XGBoost")
#     train_iter = XGBDataIterator(train_ds)
#     train_Xy = xgb.DMatrix(train_iter)
#
#     val_iter = XGBDataIterator(val_ds)
#     val_Xy = xgb.DMatrix(val_iter)
#
#     # Training
#     console.print("Running training")
#     eval_dict = {}
#     model = xgb.train(
#         model_params,
#         train_Xy,
#         feval=xgb_mse_metric,
#         num_boost_round=train_config["n_epochs"],
#         evals=[(train_Xy, "train"), (val_Xy, "val")],
#         evals_result=eval_dict,
#         callbacks=train_config["callbacks"],
#     )
#
#     # Save the model
#     model_ffp = output_dir / "xgb.model"
#     model.save_model(model_ffp)
#
#     # Generate importance plot
#     model.feature_names = list(feature_config.keys())
#     fig = plt.figure(figsize=(16, 10), dpi=200)
#     ax = fig.add_subplot(1, 1, 1)
#     xgb.plot_importance(model, ax=ax)
#     fig.savefig(output_dir / "feature_importance.png")
#     plt.close()
#
#     eval_key = "MSE"
#     fig = ml_tools.plotting.plot_loss(
#         dict(loss=eval_dict["train"][eval_key], val_loss=eval_dict["train"][eval_key]),
#         y_lim=(0.0, 1.0),
#         y_label=eval_key,
#     )
#     fig.savefig(output_dir / "loss_plot.png")
#     plt.close()
#
#     return TrainingResult(io_config, train_config, output_dir, model_ffp)
