"""Main training pipeline for ASL hand sign detection models.

Provides a unified interface for training all model architectures with
consistent callbacks, logging, and evaluation.
"""

import os
import time
import logging
import json
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .callbacks import get_callbacks, LearningRateLogger

logger = logging.getLogger(__name__)


class ASLTrainer:
    """Unified training interface for all ASL detection models.

    Handles training, validation, model saving, and history logging for
    any Keras model. Supports early stopping, LR reduction, and
    model checkpointing.

    Attributes:
        model: The Keras model to train.
        model_name: Human-readable model name.
        save_dir: Directory for saving model weights.
        log_dir: Directory for training logs.
    """

    def __init__(self, model: keras.Model,
                 model_name: str,
                 save_dir: str = "results/models",
                 log_dir: str = "results/logs",
                 seed: int = 42):
        """Initialize the trainer.

        Args:
            model: Compiled Keras model to train.
            model_name: Name for saving and logging.
            save_dir: Directory to save checkpoints.
            log_dir: Directory for training logs.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.model_name = model_name
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.seed = seed
        self.history = None
        self.training_time = None

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        tf.random.set_seed(seed)
        np.random.seed(seed)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 64,
              class_weights: Optional[Dict[int, float]] = None,
              patience_es: int = 10,
              patience_lr: int = 5,
              use_augmentation: bool = False,
              augmentor=None) -> keras.callbacks.History:
        """Train the model with full callback suite.

        Args:
            X_train: Training images.
            y_train: Training labels.
            X_val: Validation images.
            y_val: Validation labels.
            epochs: Maximum training epochs.
            batch_size: Batch size.
            class_weights: Class weight dictionary for imbalanced data.
            patience_es: Early stopping patience.
            patience_lr: LR reduction patience.
            use_augmentation: Whether to apply data augmentation.
            augmentor: ASLAugmentor instance (required if use_augmentation=True).

        Returns:
            Keras History object with training metrics.
        """
        logger.info(f"Training {self.model_name}...")
        logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}")
        logger.info(f"  Epochs: {epochs}, Batch: {batch_size}")

        callbacks = get_callbacks(
            model_name=self.model_name,
            save_dir=self.save_dir,
            log_dir=self.log_dir,
            patience_es=patience_es,
            patience_lr=patience_lr,
        )
        callbacks.append(LearningRateLogger())

        start_time = time.time()

        if use_augmentation and augmentor is not None:
            train_dataset = augmentor.create_tf_dataset(
                X_train, y_train, batch_size=batch_size, augment=True
            )
            val_dataset = augmentor.create_tf_dataset(
                X_val, y_val, batch_size=batch_size, augment=False, shuffle=False
            )
            self.history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.1f}s")

        self._save_history()
        return self.history

    def train_two_stage(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        stage1_epochs: int = 20,
                        stage2_epochs: int = 30,
                        batch_size: int = 64,
                        fine_tune_fn=None) -> keras.callbacks.History:
        """Two-stage training: frozen base then fine-tuning (for transfer learning).

        Stage 1: Train with frozen base model
        Stage 2: Fine-tune with lower learning rate

        Args:
            X_train: Training images.
            y_train: Training labels.
            X_val: Validation images.
            y_val: Validation labels.
            stage1_epochs: Epochs for stage 1 (frozen base).
            stage2_epochs: Epochs for stage 2 (fine-tuning).
            batch_size: Batch size.
            fine_tune_fn: Function to call between stages to unfreeze layers.

        Returns:
            Final History object.
        """
        logger.info(f"Stage 1: Training {self.model_name} with frozen base...")
        self.train(X_train, y_train, X_val, y_val,
                   epochs=stage1_epochs, batch_size=batch_size)

        if fine_tune_fn is not None:
            logger.info("Stage 2: Fine-tuning...")
            fine_tune_fn(self.model)
            self.train(X_train, y_train, X_val, y_val,
                       epochs=stage2_epochs, batch_size=batch_size,
                       patience_es=15)

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 batch_size: int = 64) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            X_test: Test images.
            y_test: Test labels.
            batch_size: Inference batch size.

        Returns:
            Dictionary with loss and accuracy.
        """
        results = self.model.evaluate(X_test, y_test,
                                      batch_size=batch_size, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        logger.info(f"{self.model_name} test results: {metrics}")

        # Save test metrics
        metrics_path = os.path.join(self.log_dir,
                                    f"{self.model_name}_test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Get model predictions.

        Args:
            X: Input images.
            batch_size: Inference batch size.

        Returns:
            Predicted class indices.
        """
        probs = self.model.predict(X, batch_size=batch_size, verbose=0)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Get class probability predictions.

        Args:
            X: Input images.
            batch_size: Inference batch size.

        Returns:
            Probability array of shape (N, num_classes).
        """
        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the model to disk.

        Args:
            path: Save path (default: results/models/{model_name}_final.h5).

        Returns:
            Path where the model was saved.
        """
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}_final.h5")
        self.model.save(path)
        logger.info(f"Model saved to {path}")
        return path

    def load_best_model(self) -> None:
        """Load the best checkpoint saved during training."""
        checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_best.h5")
        if os.path.exists(checkpoint_path):
            self.model = keras.models.load_model(checkpoint_path)
            logger.info(f"Loaded best model from {checkpoint_path}")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}")

    def _save_history(self) -> None:
        """Save training history to JSON."""
        if self.history is None:
            return
        history_path = os.path.join(self.log_dir,
                                    f"{self.model_name}_history.json")
        history_dict = {k: [float(v) for v in vals]
                        for k, vals in self.history.history.items()}
        history_dict["training_time_seconds"] = self.training_time

        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training results.

        Returns:
            Dictionary with training summary statistics.
        """
        if self.history is None:
            return {"status": "not trained"}

        h = self.history.history
        summary = {
            "model_name": self.model_name,
            "epochs_trained": len(h.get("val_accuracy", [])),
            "best_val_accuracy": float(max(h.get("val_accuracy", [0]))),
            "final_train_accuracy": float(h.get("accuracy", [0])[-1]),
            "final_val_loss": float(h.get("val_loss", [0])[-1]),
            "training_time_seconds": self.training_time,
        }
        return summary


def main():
    """CLI entry point for running training."""
    import argparse
    from src.data.loader import ASLDataLoader
    from src.data.augmentation import ASLAugmentor
    from src.models.custom_cnn import build_custom_cnn
    from src.utils.config import load_config
    from src.utils.logger import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Train ASL detection models")
    parser.add_argument("--model", default="custom_cnn",
                        choices=["custom_cnn", "mobilenet", "efficientnet",
                                 "skeleton_gcn", "attention_cnn"],
                        help="Model architecture to train")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    loader = ASLDataLoader(data_dir=cfg["paths"]["data_raw"])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.load_dataset()

    if args.model == "custom_cnn":
        model = build_custom_cnn()
    else:
        logger.error(f"Model {args.model} not yet wired up in CLI")
        return

    trainer = ASLTrainer(model, model_name=args.model)
    trainer.train(X_train, y_train, X_val, y_val,
                  epochs=args.epochs or cfg["models"]["custom_cnn"]["epochs"])
    trainer.evaluate(X_test, y_test)
    trainer.save_model()


if __name__ == "__main__":
    main()
