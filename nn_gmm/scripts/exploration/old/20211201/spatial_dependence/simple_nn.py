#%% Imports
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

#%% Train without location (NN)
cur_features = feature_names[:-2]
cur_X_train, cur_X_test = X_train[cur_features], X_test[cur_features]

inputs = keras.Input(cur_X_train.shape[1])
x = ml_tools.hidden_layers.selu_dropout(inputs, 16, dropout=0.1)
x = ml_tools.hidden_layers.selu_dropout(x, 16, dropout=0.1)
outputs = keras.layers.Dense(1, activation="linear")(x)

model_no_loc = keras.Model(inputs=inputs, outputs=outputs)
model_no_loc.compile("Adam", loss="mse")

history_no_loc = model_no_loc.fit(cur_X_train, y_train, batch_size=32, epochs=50, validation_data=(cur_X_test, y_test))

y_est_no_loc = model_no_loc.predict(cur_X_test).ravel()

# #%% Visualise
# fig = ml_tools.plotting.plot_loss(history.history)
# plt.show()
#
# utils.vis_results(y_test.values.ravel(), model.predict(cur_X_test).ravel(), "No Location")

#%% Train with location (NN)
cur_X_train, cur_X_test = X_train, X_test

inputs = keras.Input(cur_X_train.shape[1])
x = ml_tools.hidden_layers.selu_dropout(inputs, 16, dropout=0.1)
x = ml_tools.hidden_layers.selu_dropout(x, 16, dropout=0.1)
outputs = keras.layers.Dense(1, activation="linear")(x)

model_loc = keras.Model(inputs=inputs, outputs=outputs)
model_loc.compile("Adam", loss="mse")

history_loc = model_loc.fit(cur_X_train, y_train, batch_size=32, epochs=50, validation_data=(cur_X_test, y_test))

y_est_loc = model_loc.predict(cur_X_test).ravel()

# #%% Visualise
# fig = ml_tools.plotting.plot_loss(history.history)
# plt.show()
#
# utils.vis_results(y_test.values.ravel(), model.predict(cur_X_test).ravel(), "With Location")

#%% Compare
ml_tools.plotting.compare_loss([
    ml_tools.plotting.LossPlotData(history_no_loc.history, "No location", "b"),
    ml_tools.plotting.LossPlotData(history_loc.history, "With location", "r")
], y_lim=(0.0, 1.0))

utils.compare_results(y_test.values.ravel(), y_est_no_loc, y_est_loc)


