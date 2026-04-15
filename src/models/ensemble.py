"""Ensemble methods for combining multiple ASL detection models.

Implements:
- Weighted average ensemble
- Majority voting ensemble
- Stacking ensemble with meta-learner
"""

import logging
import os
from typing import List, Optional, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Combines predictions from multiple models using various strategies.

    Supports:
    - Weighted average of softmax probabilities
    - Majority voting on predicted classes
    - Stacking with a meta-learner

    Attributes:
        models: List of trained Keras models.
        weights: Optional weights for weighted average (must sum to 1).
        strategy: Ensemble strategy ('weighted_average', 'voting', 'stacking').
    """

    def __init__(self, models: List[keras.Model],
                 weights: Optional[List[float]] = None,
                 strategy: str = "weighted_average",
                 model_names: Optional[List[str]] = None):
        """Initialize the ensemble.

        Args:
            models: List of trained Keras models.
            weights: Optional weights for weighted average ensemble.
                     If None, equal weights are used.
            strategy: Ensemble strategy.
            model_names: Optional display names for each model.
        """
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models.")

        self.models = models
        self.strategy = strategy
        self.model_names = model_names or [f"model_{i}" for i in range(len(models))]
        self.meta_learner = None

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("weights length must match number of models.")
            total = sum(weights)
            self.weights = [w / total for w in weights]

        logger.info(
            f"Ensemble ({strategy}) created with {len(models)} models. "
            f"Weights: {[f'{w:.3f}' for w in self.weights]}"
        )

    def predict_proba(self, X: np.ndarray,
                      batch_size: int = 64) -> np.ndarray:
        """Get ensemble probability predictions.

        Args:
            X: Input data array.
            batch_size: Batch size for inference.

        Returns:
            Probability array of shape (N, num_classes).
        """
        all_probs = []
        for model in self.models:
            probs = model.predict(X, batch_size=batch_size, verbose=0)
            all_probs.append(probs)

        if self.strategy == "weighted_average":
            ensemble_probs = np.zeros_like(all_probs[0])
            for probs, weight in zip(all_probs, self.weights):
                ensemble_probs += weight * probs
            return ensemble_probs

        elif self.strategy == "voting":
            # Hard voting: each model votes for a class
            votes = np.array([np.argmax(p, axis=1) for p in all_probs])
            # Convert votes to probabilities via counting
            n_classes = all_probs[0].shape[1]
            n_samples = votes.shape[1]
            ensemble_probs = np.zeros((n_samples, n_classes))
            for i in range(n_samples):
                for vote in votes[:, i]:
                    ensemble_probs[i, vote] += 1
            ensemble_probs /= len(self.models)
            return ensemble_probs

        elif self.strategy == "stacking":
            if self.meta_learner is None:
                raise RuntimeError("Meta-learner not trained. Call fit_meta_learner first.")
            stacked = np.concatenate(all_probs, axis=1)
            return self.meta_learner.predict_proba(stacked)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Get ensemble class predictions.

        Args:
            X: Input data array.
            batch_size: Batch size for inference.

        Returns:
            Predicted class indices of shape (N,).
        """
        probs = self.predict_proba(X, batch_size)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 batch_size: int = 64) -> Dict[str, float]:
        """Evaluate ensemble on labeled data.

        Args:
            X: Input data.
            y: True labels.
            batch_size: Inference batch size.

        Returns:
            Dictionary with accuracy and per-model accuracies.
        """
        y_pred = self.predict(X, batch_size)
        ensemble_acc = np.mean(y_pred == y)

        results = {"ensemble_accuracy": float(ensemble_acc)}

        for name, model in zip(self.model_names, self.models):
            preds = np.argmax(
                model.predict(X, batch_size=batch_size, verbose=0), axis=1
            )
            acc = np.mean(preds == y)
            results[f"{name}_accuracy"] = float(acc)
            logger.info(f"  {name}: {acc:.4f}")

        logger.info(f"  Ensemble ({self.strategy}): {ensemble_acc:.4f}")
        return results

    def fit_meta_learner(self, X_val: np.ndarray, y_val: np.ndarray,
                         batch_size: int = 64) -> None:
        """Train a logistic regression meta-learner for stacking ensemble.

        Args:
            X_val: Validation data for stacking training.
            y_val: Validation labels.
            batch_size: Inference batch size.
        """
        from sklearn.linear_model import LogisticRegression

        all_probs = []
        for model in self.models:
            probs = model.predict(X_val, batch_size=batch_size, verbose=0)
            all_probs.append(probs)

        stacked = np.concatenate(all_probs, axis=1)
        self.meta_learner = LogisticRegression(
            max_iter=1000, multi_class="multinomial", random_state=42
        )
        self.meta_learner.fit(stacked, y_val)
        logger.info("Meta-learner trained for stacking ensemble.")

    def get_model_comparison(self, X: np.ndarray, y: np.ndarray,
                              batch_size: int = 64) -> Dict[str, float]:
        """Compare individual model accuracies vs. ensemble.

        Args:
            X: Test data.
            y: True labels.
            batch_size: Inference batch size.

        Returns:
            Dictionary of model_name -> accuracy.
        """
        return self.evaluate(X, y, batch_size)


def build_keras_ensemble(models: List[keras.Model],
                          num_classes: int = 24,
                          name: str = "keras_ensemble") -> keras.Model:
    """Build a Keras functional ensemble model that averages predictions.

    This creates a proper Keras model that can be saved as a single file.

    Args:
        models: List of trained Keras models.
        num_classes: Number of output classes.
        name: Model name.

    Returns:
        Keras ensemble model.
    """
    if len(models) < 2:
        raise ValueError("Need at least 2 models for ensemble.")

    # Use the first model's input shape
    inputs = models[0].input

    outputs_list = []
    for i, model in enumerate(models):
        # Freeze all base models
        model.trainable = False
        outputs_list.append(model.output)

    # Average the softmax outputs
    if len(outputs_list) == 1:
        avg = outputs_list[0]
    else:
        avg = keras.layers.Average()(outputs_list)

    ensemble = keras.Model(inputs=inputs, outputs=avg, name=name)
    ensemble.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return ensemble
