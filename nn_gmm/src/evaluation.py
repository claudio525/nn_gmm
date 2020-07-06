import pickle
from typing import Tuple

import pandas as pd
import numpy as np
from tensorflow import keras

from . import training
from .model import GMM


class EvaluationResult:
    def __init__(
        self,
        training_result: training.TrainingResult,
        mean_train_est: pd.DataFrame,
        std_train_est: pd.DataFrame,
        mean_val_est: pd.DataFrame,
        std_val_est: pd.DataFrame,
    ):
        self.training_result = training_result

        self.mean_train_est = mean_train_est
        self.std_train_est = std_train_est

        self.mean_val_est = mean_val_est
        self.std_val_est = std_val_est

    def save(self, output_ffp: str):
        with open(output_ffp, "wb") as f:
            pickle.dump(self, f)


def evaluate(
    training_result: training.TrainingResult,
    train_data: Tuple[pd.DataFrame, pd.DataFrame],
    val_data: Tuple[pd.DataFrame, pd.DataFrame],
) -> EvaluationResult:
    """Evaluates a trained model

    Parameters
    ----------
    training_result

    Returns
    -------
    EvaluationResult
    """
    print(f"============================== Evaluation ===============================")

    # Load the model
    print(f"Loading model {training_result.best_model_dir}")
    model = GMM.load(training_result.best_model_dir)

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Get training and validation dataset predictions
    print(f"Computing training and validation predictions")
    mean_train_est, std_train_est = model.predict(
        X_train, pre_process=False, result_df_index=X_train.index.values.astype(str)
    )

    mean_val_est, std_val_est = model.predict(
        X_val, pre_process=False, result_df_index=X_val.index.values.astype(str)
    )

    # # Compute residuals
    # res_train = y_train_est - training_result.y_train
    # res_val = y_val_est - training_result.y_val
    #
    # ln_res_train = (y_train_est / training_result.y_train).apply(np.log)
    # ln_res_val = (y_val_est / training_result.y_val).apply(np.log)
    #
    # rel_res_train = res_train / training_result.y_train
    # rel_res_val = res_val / training_result.y_val

    return EvaluationResult(
        training_result, mean_train_est, std_train_est, mean_val_est, std_val_est
    )
