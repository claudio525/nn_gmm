import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from nn_gmm import hidden_layers
from nn_gmm import train_utils

# Config
sample_db_ffp = "/Users/Clus/code/work/nn_gmm/data/sample_dbs/v18p6.h5"

ignore_features = ["rtvz"]
categorial_features = ["tect_type"]

n_hidden_layers = 3
units = [60, 60, 60]
dropout = 0.5

hidden_layer_config = {"dropout": 0.5}
hidden_layer_func = hidden_layers.relu_BN_dropout

val_size = 0.1
batch_size = 32
n_epochs = 10


#%% Load & clean the data
X, y = train_utils.load_clean_samples(sample_db_ffp, ignore_features, categorial_features)

#%% Split into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)
n_train, n_val = X_train.shape[0], X_val.shape[0]
n_features, n_outputs = X_train.shape[1], y_train.shape[1]

# Create the train & validation datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
train_dataset = train_dataset.shuffle(n_train).batch(batch_size).prefetch(50)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))
val_dataset = val_dataset.batch(batch_size).prefetch(50)



#%% Create the model
input = keras.Input(n_features)

x = hidden_layer_func(input, units[0], **hidden_layer_config)
for unit in units[1:]:
    x = hidden_layer_func(x, unit, **hidden_layer_config)

outputs = keras.layers.Dense(units=n_outputs, activation=None)(x)

#%% Create & compile model
model = keras.Model(inputs=input, outputs=outputs)
model.compile(optimizer="Adam", loss="MSE")

# Model architecture summary
model.summary()

#%% Train
print(f"Training...")
model.fit(train_dataset, epochs=n_epochs, validation_data=val_dataset)


exit()


