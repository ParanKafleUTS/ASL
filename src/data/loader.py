"""Data loading and preprocessing for the ASL hand sign image dataset.

This module loads the ``grassknoted/asl-alphabet`` dataset
(downloaded with ``kagglehub``) which stores images in a directory hierarchy::

    <data_dir>/
    ├── asl_alphabet_train/
    │   └── asl_alphabet_train/
    │       ├── A/
    │       │   ├── A1.jpg
    │       │   └── ...
    │       ├── B/
    │       └── ...
    └── asl_alphabet_test/   ← optional

The loader auto-detects flat, single-nested, and doubly-nested layouts and is
agnostic to the number of classes (29 for A–Z + del/nothing/space).
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Common names for the train/test split sub-folders used by ASL Kaggle datasets
_TRAIN_DIR_NAMES = ("Train", "train", "training",
                    "asl_alphabet_train", "asl_alphabet_train/asl_alphabet_train")
_TEST_DIR_NAMES = ("Test", "test", "testing",
                   "asl_alphabet_test", "asl_alphabet_test/asl_alphabet_test")

# Supported image extensions
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class ASLDataLoader:
    """Load and preprocess the ASL hand sign image dataset.

    The dataset is expected to live in a directory that contains per-class
    sub-folders (optionally nested inside a ``Train/`` / ``Test/`` split).

    Attributes:
        data_dir: Root directory of the downloaded dataset.
        image_size: Images are resized to ``(image_size, image_size)``.
        color_mode: ``"rgb"`` (default) or ``"grayscale"``.
        num_classes: Number of ASL classes detected in the dataset.
    """

    def __init__(self, data_dir: str = "data/raw",
                 image_size: int = 64,
                 color_mode: str = "rgb"):
        """Initialise the loader.

        Args:
            data_dir: Root directory that was populated by
                ``data/download_dataset.py``.
            image_size: Side length (pixels) to resize every image to.
            color_mode: ``"rgb"`` loads 3-channel images; ``"grayscale"``
                loads single-channel images.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.color_mode = color_mode.lower()

        self._class_names: Optional[List[str]] = None
        self._class_weights: Optional[Dict[int, float]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> List[str]:
        """Sorted list of class names (available after ``load_dataset``)."""
        if self._class_names is None:
            raise RuntimeError(
                "class_names is not available until load_dataset() has been called."
            )
        return self._class_names

    @property
    def num_classes(self) -> int:
        """Number of classes (available after ``load_dataset``)."""
        return len(self.class_names)

    @property
    def class_weights(self) -> Optional[Dict[int, float]]:
        """Class weights for imbalanced datasets (available after ``load_dataset``)."""
        return self._class_weights

    def get_label_name(self, label_idx: int) -> str:
        """Return the class name for a numeric label index.

        Args:
            label_idx: Integer class index.

        Returns:
            Corresponding class name string (e.g. ``"A"``).
        """
        names = self.class_names
        if 0 <= label_idx < len(names):
            return names[label_idx]
        return f"Class_{label_idx}"

    def load_dataset(self, val_split: float = 0.15
                     ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                Tuple[np.ndarray, np.ndarray],
                                Tuple[np.ndarray, np.ndarray]]:
        """Load images from the dataset directory and split into train/val/test.

        If the dataset already contains a ``Train/`` and ``Test/`` split the
        loader respects it (using ``Test/`` as the held-out test set and
        carving a validation set from ``Train/``).  Otherwise a stratified
        80 / val_split / (remaining) split is created on-the-fly.

        Args:
            val_split: Fraction of *training* data reserved for validation.

        Returns:
            A tuple ``((X_train, y_train), (X_val, y_val), (X_test, y_test))``
            where every image array has shape
            ``(N, image_size, image_size, C)`` and values in ``[0, 1]``.

        Raises:
            FileNotFoundError: If no class directories are found.
        """
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: '{self.data_dir}'. "
                "Run `python data/download_dataset.py` first."
            )

        train_root, test_root = self._locate_split_dirs()

        # Load training (and possibly all) data
        X_all, y_all = self._load_split(train_root)

        if test_root is not None:
            # Dataset ships with a dedicated test split
            X_test, y_test = self._load_split(test_root)
            X_train, X_val, y_train, y_val = train_test_split(
                X_all, y_all,
                test_size=val_split,
                stratify=y_all,
                random_state=42,
            )
        else:
            # Create val + test from the single directory
            test_fraction = 0.15
            X_tmp, X_test, y_tmp, y_test = train_test_split(
                X_all, y_all,
                test_size=test_fraction,
                stratify=y_all,
                random_state=42,
            )
            relative_val = val_split / (1.0 - test_fraction)
            X_train, X_val, y_train, y_val = train_test_split(
                X_tmp, y_tmp,
                test_size=relative_val,
                stratify=y_tmp,
                random_state=42,
            )

        self._class_weights = self._compute_class_weights(y_train)

        logger.info(
            f"Splits — Train: {X_train.shape}, Val: {X_val.shape}, "
            f"Test: {X_test.shape}"
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def visualize_samples(self, X: np.ndarray, y: np.ndarray,
                          n_samples: int = 24,
                          save_path: Optional[str] = None) -> None:
        """Show a grid of sample images with their class labels.

        Args:
            X: Image array of shape ``(N, H, W, C)``.
            y: Integer label array of shape ``(N,)``.
            n_samples: Number of images to display.
            save_path: If given, the figure is saved to this path.
        """
        n_cols = 6
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows))

        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        for ax, idx in zip(axes.flat, indices):
            img = X[idx]
            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(-1), cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(self.get_label_name(int(y[idx])), fontsize=8)
            ax.axis("off")

        for ax in axes.flat[len(indices):]:
            ax.axis("off")

        plt.suptitle("ASL Hand Sign Samples", fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Sample visualization saved to {save_path}")
        plt.show()

    def visualize_class_distribution(self, y: np.ndarray,
                                     title: str = "Class Distribution",
                                     save_path: Optional[str] = None) -> None:
        """Plot the class distribution of a dataset split.

        Args:
            y: Integer label array.
            title: Plot title.
            save_path: If given, the figure is saved to this path.
        """
        labels, counts = np.unique(y, return_counts=True)
        label_names = [self.get_label_name(int(l)) for l in labels]

        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(label_names, counts, color="steelblue",
                      edgecolor="black", alpha=0.8)
        ax.set_xlabel("ASL Sign", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=8,
            )

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _locate_split_dirs(self) -> Tuple[str, Optional[str]]:
        """Find the train-root and (optional) test-root inside *data_dir*.

        Uses ``_is_class_root`` / ``_find_class_root`` to distinguish actual
        class directories (which contain image files) from split-level
        directories (e.g. ``Train/``, ``asl_alphabet_train/``), so the logic
        works for both flat and doubly-nested dataset layouts.

        Returns:
            ``(train_root, test_root)`` — *test_root* may be ``None``.
        """
        train_root: Optional[str] = None
        test_root: Optional[str] = None

        # 1. Try named train directories first
        for name in _TRAIN_DIR_NAMES:
            candidate = os.path.join(self.data_dir, name)
            if os.path.isdir(candidate):
                root = self._find_class_root(candidate)
                if root is not None:
                    train_root = root
                    break

        # 2. Try named test directories
        for name in _TEST_DIR_NAMES:
            candidate = os.path.join(self.data_dir, name)
            if os.path.isdir(candidate):
                root = self._find_class_root(candidate)
                if root is not None:
                    test_root = root
                    break

        # 3. Fall back: search from data_dir itself (flat layout)
        if train_root is None:
            root = self._find_class_root(self.data_dir)
            if root is not None:
                train_root = root

        if train_root is None:
            raise FileNotFoundError(
                f"Could not find class subdirectories under '{self.data_dir}'. "
                "Please verify the dataset was downloaded correctly."
            )

        return train_root, test_root

    @staticmethod
    def _is_class_root(path: str) -> bool:
        """Return True if *path* is a directory whose sub-folders contain image files.

        This distinguishes a genuine class-root (e.g. ``Train/``, which has
        ``A/``, ``B/`` … each holding ``.jpg`` files) from an intermediate
        split wrapper (e.g. ``asl_alphabet_train/`` whose only child is another
        directory, not images).
        """
        if not os.path.isdir(path):
            return False
        for entry in os.listdir(path):
            if entry.startswith("."):
                continue
            sub = os.path.join(path, entry)
            if not os.path.isdir(sub):
                continue
            # A valid class dir must contain at least one image file
            for fname in os.listdir(sub):
                if os.path.splitext(fname)[1].lower() in _IMG_EXTENSIONS:
                    return True
        return False

    @staticmethod
    def _find_class_root(directory: str, max_depth: int = 4) -> Optional[str]:
        """Recursively find the first descendant directory that is a class root.

        Performs a depth-first search starting from *directory*, stopping at
        *max_depth* levels.  Returns ``None`` if no class root is found.
        """
        if ASLDataLoader._is_class_root(directory):
            return directory
        if max_depth <= 0:
            return None
        for entry in sorted(os.listdir(directory)):
            if entry.startswith("."):
                continue
            sub = os.path.join(directory, entry)
            if os.path.isdir(sub):
                result = ASLDataLoader._find_class_root(sub, max_depth - 1)
                if result is not None:
                    return result
        return None

    def _load_split(self, split_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load all images from a split directory (which contains class sub-folders).

        Args:
            split_dir: Path to a directory whose sub-folders are class names.

        Returns:
            ``(X, y)`` arrays.
        """
        class_names = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith(".")
        ])

        if not class_names:
            raise FileNotFoundError(
                f"No class subdirectories found in '{split_dir}'."
            )

        # Record class names from the first split that is loaded
        if self._class_names is None:
            self._class_names = class_names
        else:
            # Ensure consistency between train and test splits
            if set(class_names) != set(self._class_names):
                logger.warning(
                    "Class names differ between splits. "
                    f"Using names from training split: {self._class_names}"
                )
                # Map test-split names to training-split indices
                class_names = self._class_names

        logger.info(f"Loading {len(class_names)} classes from '{split_dir}'")

        all_images: List[np.ndarray] = []
        all_labels: List[int] = []

        for idx, cls in enumerate(class_names):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            imgs = self._load_images_from_dir(cls_dir)
            all_images.extend(imgs)
            all_labels.extend([idx] * len(imgs))
            logger.debug(f"  '{cls}': {len(imgs)} images")

        if not all_images:
            raise FileNotFoundError(
                f"No images found under '{split_dir}'. "
                "Check that the dataset contains supported image files "
                f"({', '.join(sorted(_IMG_EXTENSIONS))})."
            )

        X = np.array(all_images, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        logger.info(f"Loaded {len(X)} images, shape {X.shape}")
        return X, y

    def _load_images_from_dir(self, class_dir: str) -> List[np.ndarray]:
        """Load all supported image files from *class_dir*.

        Args:
            class_dir: Directory containing image files for a single class.

        Returns:
            List of float32 arrays with shape
            ``(image_size, image_size, C)`` and values in ``[0, 1]``.
        """
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to load image files. "
                "Run: pip install Pillow"
            ) from exc

        images: List[np.ndarray] = []
        for fname in sorted(os.listdir(class_dir)):
            if os.path.splitext(fname)[1].lower() not in _IMG_EXTENSIONS:
                continue
            fpath = os.path.join(class_dir, fname)
            try:
                with PILImage.open(fpath) as img:
                    img = img.convert("L" if self.color_mode == "grayscale" else "RGB")
                    img = img.resize(
                        (self.image_size, self.image_size), PILImage.BILINEAR
                    )
                    arr = np.array(img, dtype=np.float32) / 255.0
                    if self.color_mode == "grayscale":
                        arr = np.expand_dims(arr, axis=-1)  # (H, W, 1)
                images.append(arr)
            except Exception as exc:
                logger.warning(f"Skipping '{fpath}': {exc}")

        return images

    @staticmethod
    def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights to mitigate class imbalance.

        Args:
            y: Integer label array.

        Returns:
            Dict mapping class index to weight.
        """
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return dict(zip(classes.astype(int), weights))
