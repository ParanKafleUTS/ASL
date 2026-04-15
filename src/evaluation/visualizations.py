"""Visualization utilities for ASL detection results.

Creates publication-ready plots for:
- Training curves (loss and accuracy)
- Confusion matrices
- Per-class performance bars
- Model comparison plots
- Robustness analysis charts
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

logger = logging.getLogger(__name__)

# Publication-quality style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

ASL_LABELS = [chr(ord('A') + i) for i in range(26) if i not in (9, 25)]


def plot_training_history(history_dict: Dict,
                          model_name: str = "Model",
                          save_path: Optional[str] = None) -> None:
    """Plot training and validation loss/accuracy curves.

    Args:
        history_dict: Keras History.history dictionary.
        model_name: Model name for the plot title.
        save_path: Optional path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    # Accuracy plot
    ax1.plot(epochs, history_dict.get("accuracy", []), "b-o", label="Train", markersize=3)
    ax1.plot(epochs, history_dict.get("val_accuracy", []), "r-o", label="Validation",
             markersize=3)
    ax1.set_title(f"{model_name} - Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(epochs, history_dict.get("loss", []), "b-o", label="Train", markersize=3)
    ax2.plot(epochs, history_dict.get("val_loss", []), "r-o", label="Validation",
             markersize=3)
    ax2.set_title(f"{model_name} - Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           class_names: Optional[List[str]] = None,
                           model_name: str = "Model",
                           normalize: bool = True,
                           save_path: Optional[str] = None) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.
        model_name: Model name for title.
        normalize: Whether to normalize rows to show percentages.
        save_path: Optional path to save the figure.
    """
    from sklearn.metrics import confusion_matrix

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if class_names is None:
        class_names = ASL_LABELS
    display_names = [class_names[l] if l < len(class_names) else str(l) for l in labels]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0)
        fmt = ".2f"
        cbar_label = "Proportion"
    else:
        cm_plot = cm
        fmt = "d"
        cbar_label = "Count"

    fig_size = max(8, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(cm_plot, annot=len(labels) <= 30, fmt=fmt, cmap="Blues",
                xticklabels=display_names, yticklabels=display_names,
                ax=ax, cbar_kws={"label": cbar_label},
                annot_kws={"size": 8})

    ax.set_title(f"{model_name} - Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_per_class_metrics(metrics: Dict,
                            model_name: str = "Model",
                            save_path: Optional[str] = None) -> None:
    """Plot per-class precision, recall, and F1 bars.

    Args:
        metrics: Output from ASLEvaluator.compute_metrics().
        model_name: Model name for title.
        save_path: Optional save path.
    """
    per_class = metrics.get("per_class", {})
    if not per_class:
        logger.warning("No per-class metrics to plot.")
        return

    classes = list(per_class.keys())
    precision = [per_class[c]["precision"] for c in classes]
    recall = [per_class[c]["recall"] for c in classes]
    f1 = [per_class[c]["f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.5), 6))
    ax.bar(x - width, precision, width, label="Precision", alpha=0.8, color="steelblue")
    ax.bar(x, recall, width, label="Recall", alpha=0.8, color="orange")
    ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8, color="green")

    ax.set_xlabel("ASL Sign")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} - Per-Class Performance", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=metrics.get("f1_macro", 0), color="red",
               linestyle="--", alpha=0.7, label=f"Macro F1={metrics.get('f1_macro', 0):.3f}")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_model_comparison(results: Dict[str, Dict],
                           metrics_to_plot: List[str] = None,
                           save_path: Optional[str] = None) -> None:
    """Plot comparison of multiple models on key metrics.

    Args:
        results: Dictionary mapping model_name -> metrics dict.
        metrics_to_plot: List of metric names to include. Defaults to standard set.
        save_path: Optional save path.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["accuracy", "f1_macro", "f1_weighted", "precision_macro",
                           "recall_macro"]

    model_names = list(results.keys())
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for ax, metric in zip(axes, metrics_to_plot):
        values = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(model_names, values, color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_robustness_results(robustness_results: Dict[str, Dict],
                             save_path: Optional[str] = None) -> None:
    """Plot robustness test results across perturbation types.

    Args:
        robustness_results: Output from RobustnessTester.test_all().
        save_path: Optional save path.
    """
    test_names = list(robustness_results.keys())
    accuracies = [robustness_results[t]["accuracy"] for t in test_names]
    f1_scores = [robustness_results[t]["f1_macro"] for t in test_names]

    x = np.arange(len(test_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(test_names) * 1.2), 6))
    ax.bar(x - width / 2, accuracies, width, label="Accuracy", alpha=0.8, color="steelblue")
    ax.bar(x + width / 2, f1_scores, width, label="F1 Macro", alpha=0.8, color="orange")

    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel("Score")
    ax.set_title("Model Robustness Under Perturbations", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to file or display it.

    Args:
        fig: Matplotlib figure.
        save_path: Optional path to save figure.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
