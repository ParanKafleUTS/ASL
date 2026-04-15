"""Utility functions for data handling and preprocessing."""

import os
import logging
import hashlib
import pickle
from typing import Any, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, random, and TensorFlow.

    Args:
        seed: The seed value to use.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logger.debug(f"Random seed set to {seed}")


def normalize_images(images: np.ndarray, min_val: float = 0.0,
                     max_val: float = 1.0) -> np.ndarray:
    """Normalize image pixel values to a given range.

    Args:
        images: Input image array.
        min_val: Minimum output value.
        max_val: Maximum output value.

    Returns:
        Normalized image array.
    """
    imgs_min = images.min()
    imgs_max = images.max()
    if imgs_max == imgs_min:
        return np.full_like(images, min_val, dtype=np.float32)
    normalized = (images - imgs_min) / (imgs_max - imgs_min)
    normalized = normalized * (max_val - min_val) + min_val
    return normalized.astype(np.float32)


def save_pickle(obj: Any, path: str) -> None:
    """Save an object to a pickle file.

    Args:
        obj: Object to serialize.
        path: File path to save to.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {path}")


def load_pickle(path: str) -> Any:
    """Load an object from a pickle file.

    Args:
        path: File path to load from.

    Returns:
        Deserialized object.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded object from {path}")
    return obj


def compute_dataset_stats(images: np.ndarray) -> dict:
    """Compute basic statistics of a dataset.

    Args:
        images: Image array of shape (N, H, W) or (N, H, W, C).

    Returns:
        Dictionary with mean, std, min, max, shape statistics.
    """
    return {
        "shape": images.shape,
        "mean": float(images.mean()),
        "std": float(images.std()),
        "min": float(images.min()),
        "max": float(images.max()),
        "n_samples": images.shape[0],
    }


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoded format.

    Args:
        labels: Integer label array of shape (N,).
        num_classes: Total number of classes.

    Returns:
        One-hot encoded array of shape (N, num_classes).
    """
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels.astype(int)] = 1.0
    return one_hot


def resize_images(images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize images to a target size using OpenCV.

    Args:
        images: Input array of shape (N, H, W) or (N, H, W, C).
        target_size: Target (height, width).

    Returns:
        Resized image array.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for resizing. Run: pip install opencv-python")

    squeezed = images.ndim == 4 and images.shape[-1] == 1
    if squeezed:
        images = images.squeeze(-1)

    resized = np.array([
        cv2.resize(img, (target_size[1], target_size[0])) for img in images
    ])

    if squeezed:
        resized = np.expand_dims(resized, -1)

    return resized


def rgb_to_grayscale(images: np.ndarray) -> np.ndarray:
    """Convert RGB images to grayscale.

    Args:
        images: Input array of shape (N, H, W, 3).

    Returns:
        Grayscale array of shape (N, H, W, 1).
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected images of shape (N, H, W, 3)")
    grayscale = 0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]
    return np.expand_dims(grayscale, -1).astype(np.float32)
