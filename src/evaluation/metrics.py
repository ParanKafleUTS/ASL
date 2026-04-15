"""Rigorous evaluation metrics for ASL detection models.

Implements:
- Accuracy, Precision, Recall, F1-score (per-class and macro/weighted)
- Confusion matrix analysis
- Bootstrap confidence intervals
- ROC-AUC for multi-class classification
- Statistical significance testing (McNemar's test)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

ASL_LABELS = [chr(ord('A') + i) for i in range(26) if i not in (9, 25)]


class ASLEvaluator:
    """Comprehensive evaluation for ASL detection models.

    Computes standard classification metrics with bootstrap confidence
    intervals and statistical significance tests.

    Attributes:
        num_classes: Number of ASL classes.
        class_names: List of class label names.
        bootstrap_samples: Number of bootstrap samples for CI.
        confidence_level: Confidence level for intervals (default 0.95).
    """

    def __init__(self, num_classes: int = 24,
                 class_names: Optional[List[str]] = None,
                 bootstrap_samples: int = 1000,
                 confidence_level: float = 0.95):
        """Initialize the evaluator.

        Args:
            num_classes: Number of classes.
            class_names: Optional list of class names.
            bootstrap_samples: Bootstrap samples for confidence intervals.
            confidence_level: Confidence level (e.g. 0.95 for 95% CI).
        """
        self.num_classes = num_classes
        self.class_names = class_names or ASL_LABELS[:num_classes]
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level

    def compute_metrics(self, y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_proba: Optional[np.ndarray] = None) -> Dict:
        """Compute comprehensive evaluation metrics.

        Args:
            y_true: True class labels.
            y_pred: Predicted class labels.
            y_proba: Optional probability predictions for AUC computation.

        Returns:
            Dictionary containing all computed metrics.
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision_macro"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["recall_macro"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["precision_weighted"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall_weighted"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # Per-class metrics
        present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        p_per = precision_score(y_true, y_pred, average=None,
                                labels=present_labels, zero_division=0)
        r_per = recall_score(y_true, y_pred, average=None,
                             labels=present_labels, zero_division=0)
        f_per = f1_score(y_true, y_pred, average=None,
                         labels=present_labels, zero_division=0)

        metrics["per_class"] = {}
        for i, label in enumerate(present_labels):
            name = self.class_names[label] if label < len(self.class_names) else str(label)
            metrics["per_class"][name] = {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f_per[i]),
            }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=present_labels)
        metrics["confusion_matrix"] = cm.tolist()

        # ROC-AUC (requires probability predictions)
        if y_proba is not None:
            try:
                # Multi-class OvR AUC
                y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
                auc = roc_auc_score(y_true_bin, y_proba, multi_class="ovr",
                                    average="macro")
                metrics["roc_auc_macro"] = float(auc)
            except Exception as e:
                logger.warning(f"AUC computation failed: {e}")

        # Classification report (text)
        metrics["classification_report"] = classification_report(
            y_true, y_pred,
            labels=present_labels,
            target_names=[self.class_names[l] for l in present_labels],
            zero_division=0
        )

        return metrics

    def bootstrap_confidence_interval(self, y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      metric: str = "accuracy") -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval for a metric.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            metric: Metric to compute CI for ('accuracy', 'f1_macro').

        Returns:
            Tuple of (estimate, lower_ci, upper_ci).
        """
        n = len(y_true)
        alpha = 1 - self.confidence_level

        bootstrapped_scores = []
        rng = np.random.default_rng(42)

        for _ in range(self.bootstrap_samples):
            indices = rng.integers(0, n, size=n)
            y_t = y_true[indices]
            y_p = y_pred[indices]

            if metric == "accuracy":
                score = accuracy_score(y_t, y_p)
            elif metric == "f1_macro":
                score = f1_score(y_t, y_p, average="macro", zero_division=0)
            elif metric == "f1_weighted":
                score = f1_score(y_t, y_p, average="weighted", zero_division=0)
            else:
                score = accuracy_score(y_t, y_p)

            bootstrapped_scores.append(score)

        bootstrapped_scores = np.array(bootstrapped_scores)
        lower = float(np.percentile(bootstrapped_scores, 100 * alpha / 2))
        upper = float(np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2)))
        estimate = float(np.mean(bootstrapped_scores))

        return estimate, lower, upper

    def mcnemar_test(self, y_true: np.ndarray,
                     y_pred1: np.ndarray,
                     y_pred2: np.ndarray) -> Dict:
        """Perform McNemar's test to compare two models statistically.

        McNemar's test checks if there is a significant difference between
        the error patterns of two classifiers on the same test set.

        Args:
            y_true: True labels.
            y_pred1: Predictions from model 1.
            y_pred2: Predictions from model 2.

        Returns:
            Dictionary with test statistic and p-value.
        """
        # Correct/incorrect for each model
        correct1 = (y_pred1 == y_true).astype(int)
        correct2 = (y_pred2 == y_true).astype(int)

        # Build 2x2 contingency table
        # b = model1 correct, model2 wrong
        # c = model1 wrong, model2 correct
        b = np.sum((correct1 == 1) & (correct2 == 0))
        c = np.sum((correct1 == 0) & (correct2 == 1))

        # McNemar's test with continuity correction
        if b + c == 0:
            return {"statistic": 0.0, "p_value": 1.0,
                    "significant": False, "b": int(b), "c": int(c)}

        statistic = (abs(b - c) - 1) ** 2 / (b + c)

        from scipy import stats
        p_value = float(1 - stats.chi2.cdf(statistic, df=1))

        return {
            "statistic": float(statistic),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "b": int(b),
            "c": int(c),
            "interpretation": (
                "Significant difference" if p_value < 0.05
                else "No significant difference"
            )
        }

    def save_metrics(self, metrics: Dict, path: str) -> None:
        """Save metrics dictionary to a JSON file.

        Args:
            metrics: Metrics dictionary.
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Remove non-serializable objects
        save_dict = {k: v for k, v in metrics.items()
                     if k != "classification_report"}
        with open(path, "w") as f:
            json.dump(save_dict, f, indent=2)
        logger.info(f"Metrics saved to {path}")

    def print_summary(self, metrics: Dict, model_name: str = "") -> None:
        """Print a formatted metrics summary.

        Args:
            metrics: Metrics dictionary from compute_metrics.
            model_name: Optional model name for the header.
        """
        header = f"=== {model_name} Evaluation ===" if model_name else "=== Evaluation ==="
        print(f"\n{header}")
        print(f"  Accuracy:          {metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (macro):        {metrics.get('f1_macro', 0):.4f}")
        print(f"  F1 (weighted):     {metrics.get('f1_weighted', 0):.4f}")
        print(f"  Precision (macro): {metrics.get('precision_macro', 0):.4f}")
        print(f"  Recall (macro):    {metrics.get('recall_macro', 0):.4f}")
        if "roc_auc_macro" in metrics:
            print(f"  ROC-AUC (macro):  {metrics['roc_auc_macro']:.4f}")
        print()


def main():
    """CLI entry point for running evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ASL models")
    parser.add_argument("--predictions", required=True, help="Path to predictions npy file")
    parser.add_argument("--labels", required=True, help="Path to labels npy file")
    parser.add_argument("--output", default="results/metrics/eval.json")
    args = parser.parse_args()

    y_pred = np.load(args.predictions)
    y_true = np.load(args.labels)

    evaluator = ASLEvaluator()
    metrics = evaluator.compute_metrics(y_true, y_pred)
    evaluator.print_summary(metrics)
    evaluator.save_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
