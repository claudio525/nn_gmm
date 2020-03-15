import os

import tensorflow as tf

from nn_gmm import hidden_layers
from nn_gmm import training, evaluation

# Grow the GPU memory usage as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



INPUT_CONFIG = {
    "sample_db_ffp": "/Users/Clus/code/work/nn_gmm/data/sample_dbs/v18p6.h5",
    "base_output_dir": "/Users/Clus/code/work/nn_gmm/results/test",
    "output_dir": None,
    "ignore_features": ["rtvz"],
    "categorial_features": ["tect_type"],
    "std_scale_features": [
        "vs30",
        "z1p0",
        "z2p5",
        "dip",
        "rake",
        "width",
        "ztor",
        "mag",
        "rjb",
        "rrup",
        "rx",
        "ry",
    ],
    "min_max_scale_features": ["lat", "lon"],
}

# Config
TRAIN_CONFIG = {
    "model_config": {
        "hidden_layer_config": {"dropout": 0.25},
        "hidden_layer_func": hidden_layers.relu_BN_dropout,
        "units": [60, 60, 60],
    },
    "training_config": {"val_size": 0.1, "batch_size": 32, "n_epochs": 5, "loss": "MSE"},
}

if __name__ == "__main__":
    train_result, output_dir = training.run(INPUT_CONFIG, TRAIN_CONFIG)
    train_result.save(os.path.join(output_dir, "train_results.pickle"))
    eval_result = evaluation.evaluate(train_result)
    eval_result.save(os.path.join(output_dir, "eval_results.pickle"))

