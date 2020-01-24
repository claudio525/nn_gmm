import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from nn_gmm import hidden_layers
from nn_gmm import train_utils

# Config
CONFIG = {
    "sample_db_ffp": "/Users/Clus/code/work/nn_gmm/data/sample_dbs/v18p6.h5",
    "output_dir": "/Users/Clus/code/work/nn_gmm/results/test",
    "ignore_features": ["rtvz"],
    "categorial_features": ["tect_type"],
    "model_config": {
        "hidden_layer_config": {"dropout": 0.5},
        "hidden_layer_func": hidden_layers.relu_BN_dropout,
        "units": [60, 60, 60],
    },
    "training_config": {
        "val_size": 0.1,
        "batch_size": 32,
        "n_epochs": 5,
        "dropout": 0.5,
    },
}


def run(config: Dict = None):
    if config is None:
        config = CONFIG

    # Load some config
    model_config = config["model_config"]
    training_config = config["training_config"]
    batch_size, n_epochs = training_config["batch_size"], training_config["n_epochs"]

    output_dir = config["output_dir"]

    # Load & clean the data
    X, y = train_utils.load_clean_samples(
        config["sample_db_ffp"],
        config["ignore_features"],
        config["categorial_features"],
    )
    # X, y = X.iloc[:10000, :], y.iloc[:10000, :]

    # Split into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=training_config["val_size"]
    )
    n_train, n_val = X_train.shape[0], X_val.shape[0]
    n_features, n_outputs = X_train.shape[1], y_train.shape[1]

    # Preprocessing
    # TODO: Generic way to support this.. (feature selection & scaling?)

    # Create the train & validation datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    train_dataset = train_dataset.shuffle(n_train).batch(batch_size).prefetch(50)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))
    val_dataset = val_dataset.batch(batch_size).prefetch(50)

    # Create the model
    print(f"Creating model")
    model = train_utils.create_model(model_config, n_features, n_outputs)
    model.compile(optimizer="Adam", loss="MSE")

    # Model architecture summary
    model.summary()
    keras.utils.plot_model(model, to_file=os.path.join(output_dir, "model.png"))

    # Callbacks
    callbacks = [
        # Saves the best model (based on the validation loss)
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
        )
    ]

    # Train
    print(f"Training...")
    history = model.fit(train_dataset, epochs=n_epochs, validation_data=val_dataset, callbacks=callbacks)

    # Create loss plot
    plt.figure()
    plt.plot(history.epoch, history.history["loss"], label="Loss")
    plt.plot(history.epoch, history.history["val_loss"], label="Validation loss")

    plt.savefig(os.path.join(output_dir, "loss.png"))

    return


if __name__ == "__main__":
    run()
