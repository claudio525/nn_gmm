from typing import List

import numpy as np
import pandas as pd


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