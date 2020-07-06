import json
import pickle
from pathlib import Path
from typing import Union, Dict, List, Tuple

import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .training import MargNLLLoss


class GMM:
    def __init__(
        self,
        model: keras.Model,
        features: np.ndarray,
        outputs: np.ndarray,
        input_config: Dict,
        std_scaler: StandardScaler,
        min_max_scaler: MinMaxScaler,
        # std_scaler_y: StandardScaler,
        cat_columns: List[str],
    ):
        self.model = model
        self.features = features
        self.outputs = outputs
        self.input_config = input_config

        self.min_max_scaler = min_max_scaler
        self.std_scaler = std_scaler
        self.cat_columns = cat_columns

        # self.std_scaler_y = std_scaler_y

    def predict(self, X: pd.DataFrame, pre_process: bool = True, result_df_index: np.ndarray = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        std_features = self.input_config["std_scale_features"]
        cat_features = self.input_config["categorial_features"]
        min_max_features = self.input_config["min_max_scale_features"]

        if pre_process:
            # Deal with the categorial features
            X = pd.get_dummies(X, columns=cat_features)

            # If the X dataframe does not have all possible values for
            # any of the categorial features the pd.get_dummies does
            # not create those columns, hence has to be done manually here
            for cur_col in self.cat_columns:
                if cur_col not in X.columns:
                    X[cur_col] = np.zeros(X.shape[0], dtype=int)

            if len(std_features) > 0:
                X.loc[:, std_features] = self.std_scaler.transform(
                    X.loc[:, std_features]
                )

            if len(min_max_features) > 0:
                X.loc[:, min_max_features] = self.min_max_scaler.transform(
                    X.loc[:, min_max_features]
                )

        # Ensure that all the required features exist
        if not np.all(np.isin(self.features, X.columns.values.astype(str))):
            raise ValueError("Not all required features exist in the given dataframe")

        y_est = self.model.predict(X.loc[:, self.features])

        mean_df = pd.DataFrame(data=np.exp(y_est[:, :self.outputs.size]), columns=self.outputs, index=result_df_index)
        std_df = pd.DataFrame(data=y_est[:, self.outputs.size:], columns=self.outputs, index=result_df_index)

        return mean_df, std_df

    @classmethod
    def load(cls, model_dir: Union[str, Path]):
        model_dir = model_dir if isinstance(model_dir, Path) else Path(model_dir)

        model = keras.models.load_model(str(model_dir), custom_objects={"MargNLLLoss": MargNLLLoss})

        # Features (and their order)
        features = np.load(model_dir / "features.npy")
        outputs = np.load(model_dir / "outputs.npy")

        # Load the scalers
        with open(model_dir / "preprocessing.pickle", "rb") as f:
            pre_dict = pickle.load(f)

        with open(model_dir / "input_config.json", "r") as f:
            input_config = json.load(f)

        return cls(
            model,
            features,
            outputs,
            input_config,
            pre_dict["std_scaler"],
            pre_dict["min_max_scaler"],
            # pre_dict["std_scaler_y"],
            pre_dict["cat_columns"],
        )
