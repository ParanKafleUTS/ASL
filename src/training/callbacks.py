"""Custom training callbacks for ASL model training."""

import os
import logging
from typing import List, Optional

import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


def get_callbacks(model_name: str,
                  save_dir: str = "results/models",
                  log_dir: str = "results/logs",
                  patience_es: int = 10,
                  patience_lr: int = 5,
                  min_lr: float = 1e-6,
                  monitor: str = "val_accuracy") -> List[keras.callbacks.Callback]:
    """Create standard training callbacks.

    Includes:
    - ModelCheckpoint: Save best model by validation accuracy
    - EarlyStopping: Stop training if no improvement
    - ReduceLROnPlateau: Reduce learning rate on plateau
    - TensorBoard: Training logs

    Args:
        model_name: Name of the model (used in file names).
        save_dir: Directory to save model checkpoints.
        log_dir: Directory for TensorBoard logs.
        patience_es: Early stopping patience (epochs).
        patience_lr: LR reduction patience (epochs).
        min_lr: Minimum learning rate for ReduceLROnPlateau.
        monitor: Metric to monitor ('val_accuracy' or 'val_loss').

    Returns:
        List of configured Keras callbacks.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.h5")
    mode = "max" if "accuracy" in monitor else "min"

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience_es,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1
    )

    tensorboard = keras.callbacks.TensorBoard(
        log_dir=os.path.join(log_dir, model_name),
        histogram_freq=0,
        update_freq="epoch"
    )

    csv_logger = keras.callbacks.CSVLogger(
        os.path.join(log_dir, f"{model_name}_history.csv"),
        append=False
    )

    callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard, csv_logger]
    logger.info(f"Created {len(callbacks)} callbacks for {model_name}")
    return callbacks


class LearningRateLogger(keras.callbacks.Callback):
    """Callback to log the current learning rate each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        if logs is not None:
            logs["learning_rate"] = lr
        logger.debug(f"Epoch {epoch + 1}: lr = {lr:.6f}")


class GradientNormLogger(keras.callbacks.Callback):
    """Callback to log gradient norms for monitoring training stability."""

    def __init__(self, log_every: int = 10):
        super().__init__()
        self.log_every = log_every
        self._batch = 0

    def on_train_batch_end(self, batch, logs=None):
        self._batch += 1
        if self._batch % self.log_every == 0:
            pass  # Gradient logging would require custom train_step override
