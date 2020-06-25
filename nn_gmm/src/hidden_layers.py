from tensorflow import keras
from tensorflow.keras import layers


def relu_BN_dropout(input: layers.Layer, n_units: int, dropout: float = 0.5):
    x = layers.Dense(units=n_units)(input)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=dropout)(x)

    return x


def relu_dropout(input: layers.Layer, n_units: int, dropout: float = 0.5):
    x = layers.Dense(units=n_units, activation="relu")(input)
    x = layers.Dropout(rate=dropout)(x)

    return x


def selu_dropout(input: layers.Layer, n_units: int, dropout: float = 0.5):
    x = layers.Dense(
        units=n_units, activation="selu", kernel_initializer="lecun_normal"
    )(input)
    x = layers.AlphaDropout(rate=dropout)(x)

    return x
