"""Data loading and preprocessing for Sign Language MNIST dataset.

This module handles loading, normalizing, and splitting the ASL dataset.
It supports the Sign Language MNIST format (CSV files with pixel values).
"""

import os
import logging
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

# ASL alphabet labels (no J=9, no Z=25 in static signs)
ASL_LABELS = [chr(ord('A') + i) for i in range(26) if i not in (9, 25)]


class ASLDataLoader:
    """Loads and preprocesses the Sign Language MNIST dataset.

    The Sign Language MNIST dataset contains 28x28 grayscale images of ASL
    hand signs for letters A-Z (excluding J and Z which require motion).

    Attributes:
        data_dir: Path to the directory containing CSV files.
        image_size: Target image size (default 28x28).
        num_classes: Number of ASL classes (default 24).
    """

    def __init__(self, data_dir: str = "data/raw", image_size: int = 28,
                 num_classes: int = 24):
        """Initialize the data loader.

        Args:
            data_dir: Directory containing sign_mnist_train.csv and sign_mnist_test.csv.
            image_size: Size of images (assumed square).
            num_classes: Number of output classes.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()
        self._class_weights: Optional[Dict[int, float]] = None

    def load_csv(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load images and labels from a Sign Language MNIST CSV file.

        Args:
            filename: Path to the CSV file.

        Returns:
            Tuple of (images, labels) arrays.
        """
        logger.info(f"Loading data from {filename}")
        df = pd.read_csv(filename)

        labels = df["label"].values
        pixels = df.drop(columns=["label"]).values

        # Reshape to (N, 28, 28) and normalize to [0, 1]
        images = pixels.reshape(-1, self.image_size, self.image_size).astype(np.float32)
        images = images / 255.0

        logger.info(f"Loaded {len(images)} samples with {len(np.unique(labels))} classes")
        return images, labels

    def load_dataset(self, val_split: float = 0.15
                     ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                Tuple[np.ndarray, np.ndarray],
                                Tuple[np.ndarray, np.ndarray]]:
        """Load the full dataset and create train/val/test splits.

        Args:
            val_split: Fraction of training data to use for validation.

        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)).

        Raises:
            FileNotFoundError: If dataset CSV files are not found.
        """
        train_path = os.path.join(self.data_dir, "sign_mnist_train.csv")
        test_path = os.path.join(self.data_dir, "sign_mnist_test.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Dataset not found in {self.data_dir}. "
                "Run `python data/download_dataset.py` to download."
            )

        X_train_full, y_train_full = self.load_csv(train_path)
        X_test, y_test = self.load_csv(test_path)

        # Stratified split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=val_split,
            stratify=y_train_full,
            random_state=42
        )

        # Expand dims for CNN (add channel dimension)
        X_train = np.expand_dims(X_train, -1)
        X_val = np.expand_dims(X_val, -1)
        X_test = np.expand_dims(X_test, -1)

        # Compute class weights for imbalanced datasets
        self._class_weights = self._compute_class_weights(y_train)

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights to handle class imbalance.

        Args:
            y: Array of class labels.

        Returns:
            Dictionary mapping class index to weight.
        """
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return dict(zip(classes.astype(int), weights))

    @property
    def class_weights(self) -> Optional[Dict[int, float]]:
        """Return computed class weights (available after load_dataset)."""
        return self._class_weights

    def get_label_name(self, label_idx: int) -> str:
        """Convert numeric label index to ASL letter.

        Args:
            label_idx: Numeric label (0-23).

        Returns:
            Corresponding ASL letter.
        """
        if 0 <= label_idx < len(ASL_LABELS):
            return ASL_LABELS[label_idx]
        return f"Class_{label_idx}"

    def visualize_samples(self, X: np.ndarray, y: np.ndarray,
                          n_samples: int = 24, save_path: Optional[str] = None) -> None:
        """Visualize a grid of sample images with their labels.

        Args:
            X: Image array of shape (N, H, W) or (N, H, W, 1).
            y: Label array of shape (N,).
            n_samples: Number of samples to display.
            save_path: If provided, save the figure to this path.
        """
        if X.ndim == 4:
            X = X.squeeze(-1)

        n_cols = 6
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows))

        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        for ax, idx in zip(axes.flat, indices):
            ax.imshow(X[idx], cmap="gray")
            ax.set_title(self.get_label_name(y[idx]), fontsize=8)
            ax.axis("off")

        # Turn off any remaining axes
        for ax in axes.flat[len(indices):]:
            ax.axis("off")

        plt.suptitle("Sign Language MNIST Samples", fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Sample visualization saved to {save_path}")
        plt.show()

    def visualize_class_distribution(self, y: np.ndarray,
                                     title: str = "Class Distribution",
                                     save_path: Optional[str] = None) -> None:
        """Plot the class distribution of a dataset split.

        Args:
            y: Label array.
            title: Plot title.
            save_path: If provided, save the figure to this path.
        """
        labels, counts = np.unique(y, return_counts=True)
        label_names = [self.get_label_name(l) for l in labels]

        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(label_names, counts, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_xlabel("ASL Sign", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 5,
                    str(count), ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
