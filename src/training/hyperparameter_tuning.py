"""Hyperparameter tuning for ASL detection models using Keras Tuner.

Implements automated hyperparameter optimization for the custom CNN and
transfer learning models using Keras Tuner's RandomSearch and Hyperband
algorithms.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


def build_tunable_cnn(hp, input_shape: Tuple = (28, 28, 1),
                      num_classes: int = 24):
    """Build a tunable CNN model for Keras Tuner.

    Hyperparameters tuned:
    - Number of filters in each conv block
    - Dropout rate
    - Dense layer units
    - Learning rate

    Args:
        hp: Keras Tuner HyperParameters object.
        input_shape: Input image shape.
        num_classes: Number of output classes.

    Returns:
        Compiled Keras model.
    """
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Tune number of conv blocks (1-4)
    n_blocks = hp.Int("n_conv_blocks", min_value=2, max_value=4, step=1)
    filter_options = [32, 64, 128]

    for i in range(n_blocks):
        n_filters = hp.Choice(f"filters_{i}", values=[32, 64, 128])
        x = layers.Conv2D(n_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        dropout_rate = hp.Float(f"conv_dropout_{i}", min_value=0.1,
                                max_value=0.5, step=0.1)
        x = layers.Dropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Tune dense layers
    n_dense = hp.Int("n_dense_layers", min_value=1, max_value=3, step=1)
    for j in range(n_dense):
        units = hp.Choice(f"dense_units_{j}", values=[64, 128, 256, 512])
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        dense_dropout = hp.Float(f"dense_dropout_{j}", min_value=0.2,
                                 max_value=0.6, step=0.1)
        x = layers.Dropout(dense_dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                  sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def run_hyperparameter_search(X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: np.ndarray,
                               y_val: np.ndarray,
                               model_builder=None,
                               project_name: str = "asl_cnn_tuning",
                               directory: str = "results/hp_tuning",
                               max_trials: int = 20,
                               epochs_per_trial: int = 20,
                               seed: int = 42) -> Dict[str, Any]:
    """Run hyperparameter search using Keras Tuner RandomSearch.

    Args:
        X_train: Training data.
        y_train: Training labels.
        X_val: Validation data.
        y_val: Validation labels.
        model_builder: Function(hp) -> keras.Model. Defaults to build_tunable_cnn.
        project_name: Name for the tuning project.
        directory: Directory to store tuning results.
        max_trials: Maximum number of hyperparameter trials.
        epochs_per_trial: Epochs to train each trial.
        seed: Random seed.

    Returns:
        Dictionary with best hyperparameters and results.
    """
    try:
        import keras_tuner as kt
    except ImportError:
        raise ImportError(
            "keras-tuner not installed. Run: pip install keras-tuner"
        )

    if model_builder is None:
        input_shape = X_train.shape[1:]
        model_builder = lambda hp: build_tunable_cnn(hp, input_shape=input_shape)

    os.makedirs(directory, exist_ok=True)

    tuner = kt.RandomSearch(
        model_builder,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=1,
        directory=directory,
        project_name=project_name,
        seed=seed,
        overwrite=False
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    logger.info(f"Starting hyperparameter search: {max_trials} trials")
    tuner.search(
        X_train, y_train,
        epochs=epochs_per_trial,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    logger.info("Best hyperparameters found:")
    for hp_name, hp_value in best_hps.values.items():
        logger.info(f"  {hp_name}: {hp_value}")

    # Evaluate best model on validation set
    val_results = best_model.evaluate(X_val, y_val, verbose=0)
    val_accuracy = val_results[1] if len(val_results) > 1 else val_results[0]
    logger.info(f"Best validation accuracy: {val_accuracy:.4f}")

    return {
        "best_hyperparameters": best_hps.values,
        "best_val_accuracy": float(val_accuracy),
        "best_model": best_model,
        "tuner": tuner
    }


def run_hyperband_search(X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          max_epochs: int = 50,
                          project_name: str = "asl_hyperband",
                          directory: str = "results/hp_tuning",
                          seed: int = 42) -> Dict[str, Any]:
    """Run Hyperband hyperparameter search for faster optimization.

    Hyperband is more efficient than random search for large search spaces.

    Args:
        X_train: Training data.
        y_train: Training labels.
        X_val: Validation data.
        y_val: Validation labels.
        max_epochs: Maximum epochs per trial.
        project_name: Project name for logging.
        directory: Directory for tuning results.
        seed: Random seed.

    Returns:
        Dictionary with best hyperparameters and model.
    """
    try:
        import keras_tuner as kt
    except ImportError:
        raise ImportError("keras-tuner not installed. Run: pip install keras-tuner")

    input_shape = X_train.shape[1:]
    model_builder = lambda hp: build_tunable_cnn(hp, input_shape=input_shape)

    os.makedirs(directory, exist_ok=True)

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=max_epochs,
        factor=3,
        directory=directory,
        project_name=project_name,
        seed=seed,
        overwrite=False
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5
    )

    logger.info(f"Starting Hyperband search, max_epochs={max_epochs}")
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    val_results = best_model.evaluate(X_val, y_val, verbose=0)
    val_accuracy = val_results[1] if len(val_results) > 1 else val_results[0]

    logger.info(f"Hyperband best val accuracy: {val_accuracy:.4f}")
    return {
        "best_hyperparameters": best_hps.values,
        "best_val_accuracy": float(val_accuracy),
        "best_model": best_model,
        "tuner": tuner
    }
