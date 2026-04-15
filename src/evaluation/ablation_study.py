"""Ablation study for ASL detection system.

Systematically removes or disables components to measure their individual
contribution to the overall model performance.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tensorflow import keras

from .metrics import ASLEvaluator

logger = logging.getLogger(__name__)


class AblationStudy:
    """Conduct systematic ablation studies on ASL detection models.

    Compares full model performance against variants with components removed
    to quantify each component's contribution.

    Attributes:
        evaluator: ASLEvaluator instance for metric computation.
        results: Dictionary storing ablation results.
    """

    def __init__(self, num_classes: int = 24):
        """Initialize the ablation study.

        Args:
            num_classes: Number of ASL classes.
        """
        self.evaluator = ASLEvaluator(num_classes=num_classes)
        self.results: Dict[str, Dict] = {}

    def run_model_comparison(self,
                              model_configs: List[Dict],
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_val: np.ndarray,
                              y_val: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              epochs: int = 30) -> Dict[str, Dict]:
        """Run ablation by comparing multiple model configurations.

        Args:
            model_configs: List of dicts, each with 'name' and 'builder' keys.
                           'builder' is a callable that returns a keras.Model.
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            X_test, y_test: Test data.
            epochs: Training epochs per configuration.

        Returns:
            Dictionary mapping config name to metrics.
        """
        from ..training.trainer import ASLTrainer

        for config in model_configs:
            name = config["name"]
            logger.info(f"Running ablation config: {name}")

            model = config["builder"]()
            trainer = ASLTrainer(model, model_name=f"ablation_{name}")
            trainer.train(X_train, y_train, X_val, y_val,
                          epochs=epochs, patience_es=10)

            y_pred = trainer.predict(X_test)
            y_proba = trainer.predict_proba(X_test)
            metrics = self.evaluator.compute_metrics(y_test, y_pred, y_proba)

            self.results[name] = {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "training_summary": trainer.get_summary(),
            }

            logger.info(
                f"  {name}: acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}"
            )

        return self.results

    def run_component_ablation(self,
                                full_model_builder,
                                ablated_builders: Dict[str, callable],
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                epochs: int = 30) -> Dict[str, Dict]:
        """Compare full model with ablated variants.

        Args:
            full_model_builder: Callable returning the full model.
            ablated_builders: Dict mapping ablation name to builder callable.
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            X_test, y_test: Test data.
            epochs: Training epochs.

        Returns:
            Results dictionary with full and ablated performance.
        """
        all_configs = [{"name": "full_model", "builder": full_model_builder}]
        for name, builder in ablated_builders.items():
            all_configs.append({"name": name, "builder": builder})

        return self.run_model_comparison(
            all_configs, X_train, y_train, X_val, y_val, X_test, y_test, epochs
        )

    def get_component_importance(self) -> Dict[str, float]:
        """Compute relative importance of each ablated component.

        Returns:
            Dictionary mapping component name to accuracy drop (positive = important).
        """
        if "full_model" not in self.results:
            logger.warning("Full model results not found. Run ablation first.")
            return {}

        full_acc = self.results["full_model"]["accuracy"]
        importance = {}

        for name, result in self.results.items():
            if name != "full_model":
                drop = full_acc - result["accuracy"]
                importance[name] = round(drop, 4)

        return importance

    def save_results(self, path: str = "results/metrics/ablation_results.json") -> None:
        """Save ablation results to JSON.

        Args:
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Ablation results saved to {path}")

    def print_summary(self) -> None:
        """Print a formatted ablation study summary."""
        print("\n=== Ablation Study Results ===")
        print(f"{'Configuration':<30} {'Accuracy':>10} {'F1 (macro)':>12}")
        print("-" * 55)

        for name, result in self.results.items():
            print(f"{name:<30} {result['accuracy']:>10.4f} {result['f1_macro']:>12.4f}")

        importance = self.get_component_importance()
        if importance:
            print("\n--- Component Importance (Accuracy Drop) ---")
            for comp, drop in sorted(importance.items(), key=lambda x: -x[1]):
                symbol = "+" if drop > 0 else "-"
                print(f"  {comp}: {symbol}{abs(drop):.4f}")
