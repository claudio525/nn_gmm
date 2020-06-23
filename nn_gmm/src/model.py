import json
import pickle
from pathlib import Path
from typing import Union, Dict

import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class GMM:
    def __init__(
        self,
        model: keras.Model,
        features: np.ndarray,
        input_config: Dict,
        std_scaler: StandardScaler,
        min_max_scaler: MinMaxScaler,
        std_scaler_y: StandardScaler,
    ):
        self.model = model
        self.features = features
        self.input_config = input_config

        self.min_max_scaler = min_max_scaler
        self.std_scaler = std_scaler

        self.std_scaler_y = std_scaler_y

    def predict(self, X: pd.DataFrame, pre_process: bool = True):
        std_features = self.input_config["std_scale_features"]
        cat_features = self.input_config["categorial_features"]
        min_max_features = self.input_config["min_max_scale_features"]

        # Ensure that all the required features exist
        if not np.all(np.isin(self.features, X.columns.values.astype(str))):
            raise ValueError("Not all required features exist in the given dataframe")

        if pre_process:
            # Deal with the categorial features
            X = pd.get_dummies(X, columns=X["categorial_features"])

            if len(std_features) > 0:
                X.loc[:, std_features] = self.std_scaler.transform(
                    X.loc[:, std_features]
                )

            if len(min_max_features) > 0:
                X.loc[:, min_max_features] = self.min_max_scaler.transform(
                    X.loc[:, min_max_features]
                )

        y_est = self.model.predict(X.loc[:, self.features])
        return self.std_scaler_y.inverse_transform(y_est)

    @classmethod
    def load(cls, model_dir: Union[str, Path]):
        model_dir = model_dir if isinstance(model_dir, Path) else Path(model_dir)

        model = keras.models.load_model(str(model_dir))

        # Features (and their order)
        features = np.load(model_dir / "features.npy")

        # Load the scalers
        with open(model_dir / "scalers.pickle", "rb") as f:
            scaler_dict = pickle.load(f)

        with open(model_dir / "input_config.json", "r") as f:
            input_config = json.load(f)

        return cls(
            model,
            features,
            input_config,
            scaler_dict["std_scaler"],
            scaler_dict["min_max_scaler"],
            scaler_dict["std_scaler_y"],
        )
