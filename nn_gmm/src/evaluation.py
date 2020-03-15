import pickle

import pandas as pd
import numpy as np
from tensorflow import keras

from . import training


class EvaluationResult:
    def __init__(
        self,
        training_result: training.TrainingResult,
        y_train_est: pd.DataFrame,
        y_val_est: pd.DataFrame,
        res_train: pd.DataFrame,
        res_val: pd.DataFrame,
        ln_res_train: pd.DataFrame,
        ln_res_val: pd.DataFrame,
        rel_res_train: pd.DataFrame,
        rel_res_val: pd.DataFrame,
    ):
        self.training_result = training_result

        self.y_train_est = y_train_est
        self.y_val_est = y_val_est

        self.res_train = res_train
        self.res_val = res_val

        self.ln_res_train = ln_res_train
        self.ln_res_val = ln_res_val

        self.rel_res_train = rel_res_train
        self.rel_res_val = rel_res_val

    def save(self, output_ffp: str):
        with open(output_ffp, "wb") as f:
            pickle.dump(self, f)




def evaluate(training_result: training.TrainingResult) -> EvaluationResult:
    """
    Evaluates a trained model

    Parameters
    ----------
    training_result

    Returns
    -------
    EvaluationResult
    """
    print(
        f"============================== Evaluation ==============================="
    )

    # Load the model
    print(f"Loading model {training_result.best_model_ffp}")
    model = keras.models.load_model(training_result.best_model_ffp)

    # Get training and validation dataset predictions
    print(f"Computing training and validation predictions")
    y_train_est = model.predict(training_result.X_train)
    y_train_est = pd.DataFrame(
        data=y_train_est,
        index=training_result.y_train.index,
        columns=training_result.y_train.columns,
    )

    y_val_est = model.predict(training_result.X_val)
    y_val_est = pd.DataFrame(
        data=y_val_est,
        index=training_result.y_val.index,
        columns=training_result.y_val.columns,
    )

    # Compute residuals
    res_train = y_train_est - training_result.y_train
    res_val = y_val_est - training_result.y_val

    ln_res_train = (y_train_est / training_result.y_train).apply(np.log)
    ln_res_val = (y_val_est / training_result.y_val).apply(np.log)

    rel_res_train = res_train / training_result.y_train
    rel_res_val = res_val / training_result.y_val

    return EvaluationResult(
        training_result,
        y_train_est,
        y_val_est,
        res_train,
        res_val,
        ln_res_train,
        ln_res_val,
        rel_res_train,
        rel_res_val,
    )
