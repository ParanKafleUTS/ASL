"""Custom CNN baseline model for ASL hand sign detection.

Architecture: 3 convolutional blocks with batch normalization, max pooling,
and dropout, followed by dense classification layers.
"""

import logging
from typing import List, Tuple, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def build_custom_cnn(input_shape: Tuple[int, int, int] = (28, 28, 1),
                     num_classes: int = 24,
                     filters: List[int] = None,
                     kernel_size: int = 3,
                     dropout_rate: float = 0.5,
                     dense_units: List[int] = None,
                     learning_rate: float = 0.001,
                     name: str = "custom_cnn") -> keras.Model:
    """Build a custom CNN baseline model.

    The architecture consists of 3 convolutional blocks:
    - Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
    Followed by global average pooling and dense layers.

    Args:
        input_shape: Shape of input images (H, W, C).
        num_classes: Number of output classes.
        filters: Number of filters in each conv block. Defaults to [32, 64, 128].
        kernel_size: Convolution kernel size.
        dropout_rate: Dropout probability for regularization.
        dense_units: Units in dense layers. Defaults to [256, 128].
        learning_rate: Adam optimizer learning rate.
        name: Model name.

    Returns:
        Compiled Keras model.
    """
    if filters is None:
        filters = [32, 64, 128]
    if dense_units is None:
        dense_units = [256, 128]

    inputs = keras.Input(shape=input_shape, name="image_input")
    x = inputs

    # Convolutional blocks
    for i, n_filters in enumerate(filters):
        x = layers.Conv2D(n_filters, kernel_size, padding="same",
                          name=f"conv_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.MaxPooling2D(2, name=f"pool_{i+1}")(x)
        x = layers.Dropout(dropout_rate * 0.5, name=f"dropout_conv_{i+1}")(x)

    # Global average pooling (better generalization than flatten)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense classification layers
    for j, units in enumerate(dense_units):
        x = layers.Dense(units, name=f"dense_{j+1}")(x)
        x = layers.BatchNormalization(name=f"bn_dense_{j+1}")(x)
        x = layers.Activation("relu", name=f"relu_dense_{j+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_dense_{j+1}")(x)

    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = keras.Model(inputs, outputs, name=name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info(f"Built {name}: {model.count_params():,} parameters")
    model.summary(print_fn=logger.debug)
    return model


def build_deeper_cnn(input_shape: Tuple[int, int, int] = (28, 28, 1),
                     num_classes: int = 24,
                     learning_rate: float = 0.001) -> keras.Model:
    """Build a deeper CNN with residual-like connections for improved accuracy.

    Args:
        input_shape: Shape of input images.
        num_classes: Number of output classes.
        learning_rate: Optimizer learning rate.

    Returns:
        Compiled Keras model.
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="deeper_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info(f"Built deeper_cnn: {model.count_params():,} parameters")
    return model
