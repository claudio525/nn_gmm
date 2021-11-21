#%% Imports
import xgboost as xgb
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import nn_gmm.scripts.exploration.spatial_dependence.utils as utils

# Grow the GPU memory usage as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import ml_tools

pd.set_option('display.max_columns', None)


#%% Load the data
data_dict = fetch_california_housing()
X, y = data_dict["data"], data_dict["target"]
feature_names = data_dict["feature_names"]

X = pd.DataFrame(data=X, columns=feature_names)
y = pd.DataFrame(data=y, columns=data_dict["target_names"])

#%% Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y)

std_scaler = StandardScaler()
X_train = pd.DataFrame(data=std_scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(data=std_scaler.transform(X_test), columns=X_test.columns)

#%% Training without location
params = {"objective": "reg:squarederror", "verbosity": 0}
cur_features = feature_names[:-2]
cur_X_train, cur_X_test = X_train[cur_features], X_test[cur_features]

cur_dtrain = xgb.DMatrix(cur_X_train, label=y_train)
cur_dtest = xgb.DMatrix(cur_X_test, label=y_test)

bst_no_loc = xgb.train(params, cur_dtrain, num_boost_round=100, evals=[(cur_dtest, "val"), (cur_dtrain, "train")])

y_est_train_no_loc = bst_no_loc.predict(cur_dtrain)
y_est_test_no_loc = bst_no_loc.predict(cur_dtest)

print(f"No-location, Training MSE: {mean_squared_error(y_train.values.ravel(), y_est_train_no_loc):.3f}")
print(f"No-location, Validation MSE: {mean_squared_error(y_test.values.ravel(), y_est_test_no_loc):.3f}")


#%% Training with location
params = {"objective": "reg:squarederror", "verbosity": 0}
cur_X_train, cur_X_test = X_train, X_test

cur_dtrain = xgb.DMatrix(cur_X_train, label=y_train)
cur_dtest = xgb.DMatrix(cur_X_test, label=y_test)

bst_loc = xgb.train(params, cur_dtrain, num_boost_round=100, evals=[(cur_dtest, "val"), (cur_dtrain, "train")])

y_est_train_loc = bst_loc.predict(cur_dtrain)
y_est_test_loc = bst_loc.predict(cur_dtest)

print(f"With-location, Training MSE: {mean_squared_error(y_train.values.ravel(), y_est_train_loc):.3f}")
print(f"With-location, Validation MSE: {mean_squared_error(y_test.values.ravel(), y_est_test_loc):.3f}")

#%% Comparison
utils.compare_results(y_test, y_est_test_no_loc, y_est_test_loc)