"""Data augmentation utilities for ASL hand sign images.

Important: Horizontal flipping is disabled as it would alter the meaning
of ASL signs. Only geometric augmentations that preserve sign semantics
are applied.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ASLAugmentor:
    """Real-time data augmentation pipeline for ASL images.

    Applies conservative augmentations that preserve the semantic meaning
    of ASL hand signs. Horizontal flipping is intentionally disabled.

    Attributes:
        rotation_range: Max rotation angle in degrees.
        zoom_range: Max zoom fraction.
        width_shift_range: Max horizontal shift as fraction of width.
        height_shift_range: Max vertical shift as fraction of height.
    """

    def __init__(self, rotation_range: float = 10.0,
                 zoom_range: float = 0.1,
                 width_shift_range: float = 0.1,
                 height_shift_range: float = 0.1):
        """Initialize the augmentor with transformation parameters.

        Args:
            rotation_range: Max rotation angle in degrees (±).
            zoom_range: Max zoom fraction (0.0 to 1.0).
            width_shift_range: Horizontal shift as fraction of width.
            height_shift_range: Vertical shift as fraction of height.
        """
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range

    def build_keras_generator(self) -> keras.preprocessing.image.ImageDataGenerator:
        """Build a Keras ImageDataGenerator with the configured augmentation.

        Returns:
            Configured ImageDataGenerator for training.
        """
        generator = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=self.rotation_range,
            zoom_range=self.zoom_range,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            horizontal_flip=False,  # Never flip ASL signs
            vertical_flip=False,
            fill_mode="nearest"
        )
        logger.info("Keras ImageDataGenerator created (no horizontal flip).")
        return generator

    def build_tf_augmentation_layer(self) -> keras.Sequential:
        """Build a TensorFlow preprocessing layer sequence for augmentation.

        Returns:
            Sequential model of augmentation layers suitable for use in a
            tf.data pipeline.
        """
        augmentation_layers = keras.Sequential([
            keras.layers.RandomRotation(
                self.rotation_range / 360.0, fill_mode="nearest"
            ),
            keras.layers.RandomZoom(
                height_factor=(-self.zoom_range, self.zoom_range),
                fill_mode="nearest"
            ),
            keras.layers.RandomTranslation(
                height_factor=self.height_shift_range,
                width_factor=self.width_shift_range,
                fill_mode="nearest"
            ),
        ], name="asl_augmentation")
        return augmentation_layers

    def create_tf_dataset(self, X: np.ndarray, y: np.ndarray,
                          batch_size: int = 64,
                          augment: bool = True,
                          shuffle: bool = True) -> tf.data.Dataset:
        """Create a tf.data.Dataset with optional augmentation.

        Args:
            X: Image array of shape (N, H, W, 1).
            y: Label array of shape (N,).
            batch_size: Batch size.
            augment: Whether to apply data augmentation.
            shuffle: Whether to shuffle the dataset.

        Returns:
            tf.data.Dataset ready for model training.
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X), seed=42)

        dataset = dataset.batch(batch_size)

        if augment:
            aug_layer = self.build_tf_augmentation_layer()

            def apply_augmentation(images, labels):
                images = aug_layer(images, training=True)
                return images, labels

            dataset = dataset.map(apply_augmentation,
                                  num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def augment_batch(self, X_batch: np.ndarray, seed: int = None) -> np.ndarray:
        """Apply augmentation to a single batch of images (NumPy-based).

        Args:
            X_batch: Batch of images (N, H, W, 1).
            seed: Optional random seed for reproducibility.

        Returns:
            Augmented batch of same shape.
        """
        if seed is not None:
            np.random.seed(seed)

        rng = np.random.default_rng(seed)
        augmented = X_batch.copy()
        n, h, w, c = augmented.shape

        for i in range(n):
            img = augmented[i, :, :, 0]

            # Random rotation
            angle = rng.uniform(-self.rotation_range, self.rotation_range)
            img = self._rotate(img, angle)

            # Random zoom
            zoom = rng.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            img = self._zoom(img, zoom)

            # Random shift
            dx = rng.uniform(-self.width_shift_range, self.width_shift_range) * w
            dy = rng.uniform(-self.height_shift_range, self.height_shift_range) * h
            img = self._shift(img, int(dx), int(dy))

            augmented[i, :, :, 0] = img

        return augmented

    @staticmethod
    def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate image by given angle using OpenCV if available, else skip."""
        try:
            import cv2
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        except ImportError:
            return image

    @staticmethod
    def _zoom(image: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Zoom/crop image to simulate zoom effect."""
        try:
            import cv2
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(image, (new_w, new_h))
            if zoom_factor > 1:
                # Crop center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                return resized[start_h:start_h + h, start_w:start_w + w]
            else:
                # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                result = np.zeros_like(image)
                result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
                return result
        except ImportError:
            return image

    @staticmethod
    def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Shift image by (dx, dy) pixels."""
        try:
            import cv2
            h, w = image.shape[:2]
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        except ImportError:
            return image
