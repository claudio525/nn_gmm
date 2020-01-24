from typing import List, Dict

import numpy as np
import pandas as pd
from tensorflow import keras


def load_clean_samples(sample_db_ffp: str, ignore_features: List[str] = None, categorial_features: List[str] = None):
    # Load the data
    with pd.HDFStore(sample_db_ffp, mode="r") as store:
        X = store["X"]
        y = store["y"]

    # Drop the rtvz columns
    if ignore_features is not None:
        X = X.drop(columns=ignore_features)

    # One hot encoding of categorial features
    if categorial_features is not None:
        if np.isin(categorial_features, X.columns):
            X = pd.get_dummies(X, columns=categorial_features)

    return X, y

def create_model(model_config: Dict, n_inputs: int, n_outputs: int):
    hidden_layer_func = model_config["hidden_layer_func"]
    hidden_layer_config = model_config["hidden_layer_config"]
    units = model_config["units"]

    input = keras.Input(n_inputs)

    x = hidden_layer_func(input, units[0], **hidden_layer_config)
    for unit in units[1:]:
        x = hidden_layer_func(x, unit, **hidden_layer_config)

    outputs = keras.layers.Dense(units=n_outputs, activation=None)(x)

    return keras.Model(inputs=input, outputs=outputs)