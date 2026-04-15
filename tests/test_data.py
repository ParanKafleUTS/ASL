"""Unit tests for data loading and preprocessing modules."""

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestASLDataUtils(unittest.TestCase):
    """Tests for src/data/utils.py"""

    def test_normalize_images_basic(self):
        """Test basic image normalization to [0, 1]."""
        from src.data.utils import normalize_images
        images = np.array([[[0, 128, 255]]], dtype=np.float32)
        normalized = normalize_images(images)
        self.assertAlmostEqual(float(normalized.min()), 0.0, places=5)
        self.assertAlmostEqual(float(normalized.max()), 1.0, places=5)

    def test_normalize_images_custom_range(self):
        """Test normalization to custom range [-1, 1]."""
        from src.data.utils import normalize_images
        images = np.arange(10, dtype=np.float32).reshape(2, 5)
        normalized = normalize_images(images, min_val=-1.0, max_val=1.0)
        self.assertAlmostEqual(float(normalized.min()), -1.0, places=5)
        self.assertAlmostEqual(float(normalized.max()), 1.0, places=5)

    def test_normalize_constant_image(self):
        """Test normalization of constant-value image (no division by zero)."""
        from src.data.utils import normalize_images
        images = np.ones((5, 5), dtype=np.float32) * 128
        normalized = normalize_images(images)
        self.assertEqual(float(normalized.min()), 0.0)

    def test_one_hot_encode(self):
        """Test one-hot encoding."""
        from src.data.utils import one_hot_encode
        labels = np.array([0, 1, 2, 0])
        one_hot = one_hot_encode(labels, num_classes=3)
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.float32)
        np.testing.assert_array_equal(one_hot, expected)

    def test_compute_dataset_stats(self):
        """Test dataset statistics computation."""
        from src.data.utils import compute_dataset_stats
        images = np.arange(100, dtype=np.float32).reshape(10, 10)
        stats = compute_dataset_stats(images)
        self.assertEqual(stats["n_samples"], 10)
        self.assertAlmostEqual(stats["min"], 0.0)
        self.assertAlmostEqual(stats["max"], 99.0)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)

    def test_set_seed_reproducibility(self):
        """Test that set_seed ensures reproducible random numbers."""
        from src.data.utils import set_seed
        set_seed(42)
        arr1 = np.random.rand(5)
        set_seed(42)
        arr2 = np.random.rand(5)
        np.testing.assert_array_equal(arr1, arr2)

    def test_save_and_load_pickle(self):
        """Test pickle save and load roundtrip."""
        from src.data.utils import save_pickle, load_pickle
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pkl")
            obj = {"key": [1, 2, 3], "value": "test"}
            save_pickle(obj, path)
            loaded = load_pickle(path)
            self.assertEqual(loaded, obj)

    def test_rgb_to_grayscale(self):
        """Test RGB to grayscale conversion."""
        from src.data.utils import rgb_to_grayscale
        # Create a simple RGB image
        images = np.ones((4, 8, 8, 3), dtype=np.float32)
        gray = rgb_to_grayscale(images)
        self.assertEqual(gray.shape, (4, 8, 8, 1))

    def test_rgb_to_grayscale_wrong_channels(self):
        """Test that wrong channel count raises ValueError."""
        from src.data.utils import rgb_to_grayscale
        images = np.ones((4, 8, 8, 1))
        with self.assertRaises(ValueError):
            rgb_to_grayscale(images)


class TestASLDataLoader(unittest.TestCase):
    """Tests for src/data/loader.py (using mock image-folder data)."""

    def _create_mock_dataset(self, tmpdir: str,
                              split_name: str = "",
                              class_names: tuple = ("A", "B", "C"),
                              n_per_class: int = 10,
                              img_size: int = 16) -> str:
        """Create a mock dataset directory with PNG images per class.

        Args:
            tmpdir: Temporary root directory.
            split_name: Sub-folder name (e.g. "Train"), or "" for flat structure.
            class_names: Sequence of class folder names.
            n_per_class: Number of images per class.
            img_size: Side length of generated images (pixels).

        Returns:
            Path to the directory containing class sub-folders.
        """
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")

        root = os.path.join(tmpdir, split_name) if split_name else tmpdir
        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(n_per_class):
                arr = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
                Image.fromarray(arr, mode="RGB").save(
                    os.path.join(cls_dir, f"img_{i:04d}.png")
                )
        return root

    def test_load_dataset_flat_structure(self):
        """load_dataset works when class folders sit directly in data_dir."""
        from src.data.loader import ASLDataLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_dataset(tmpdir, split_name="",
                                      class_names=("A", "B", "C", "D"),
                                      n_per_class=20)
            loader = ASLDataLoader(data_dir=tmpdir, image_size=16, color_mode="rgb")
            (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = loader.load_dataset(val_split=0.15)

            self.assertEqual(X_tr.ndim, 4)           # (N, 16, 16, 3)
            self.assertEqual(X_tr.shape[1], 16)
            self.assertEqual(X_tr.shape[3], 3)
            self.assertTrue(X_tr.max() <= 1.0)
            self.assertTrue(X_tr.min() >= 0.0)

    def test_load_dataset_train_test_split(self):
        """load_dataset respects an existing Train/Test directory split."""
        from src.data.loader import ASLDataLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_dataset(tmpdir, split_name="Train",
                                      class_names=("A", "B", "C"),
                                      n_per_class=20)
            self._create_mock_dataset(tmpdir, split_name="Test",
                                      class_names=("A", "B", "C"),
                                      n_per_class=5)
            loader = ASLDataLoader(data_dir=tmpdir, image_size=16)
            (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = loader.load_dataset()

            # Test set should come from the dedicated Test/ folder
            self.assertEqual(X_te.shape[0], 15)  # 3 classes × 5 images
            self.assertEqual(X_tr.ndim, 4)

    def test_class_names_detected(self):
        """class_names is populated correctly after load_dataset."""
        from src.data.loader import ASLDataLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_dataset(tmpdir, class_names=("A", "B", "C"), n_per_class=10)
            loader = ASLDataLoader(data_dir=tmpdir, image_size=16)
            loader.load_dataset()
            self.assertEqual(loader.class_names, ["A", "B", "C"])
            self.assertEqual(loader.num_classes, 3)

    def test_class_weights_computed(self):
        """class_weights is populated after load_dataset."""
        from src.data.loader import ASLDataLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_dataset(tmpdir, class_names=("A", "B", "C"), n_per_class=15)
            loader = ASLDataLoader(data_dir=tmpdir, image_size=16)
            loader.load_dataset()
            self.assertIsNotNone(loader.class_weights)
            self.assertIsInstance(loader.class_weights, dict)

    def test_missing_dir_raises(self):
        """FileNotFoundError is raised when data_dir does not exist."""
        from src.data.loader import ASLDataLoader
        loader = ASLDataLoader(data_dir="/nonexistent/path")
        with self.assertRaises(FileNotFoundError):
            loader.load_dataset()

    def test_get_label_name(self):
        """get_label_name returns the correct class string."""
        from src.data.loader import ASLDataLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_dataset(tmpdir, class_names=("A", "B", "C"), n_per_class=5)
            loader = ASLDataLoader(data_dir=tmpdir, image_size=8)
            loader.load_dataset()
            self.assertEqual(loader.get_label_name(0), "A")
            self.assertEqual(loader.get_label_name(1), "B")
            self.assertEqual(loader.get_label_name(2), "C")

    def test_grayscale_mode(self):
        """color_mode='grayscale' produces single-channel images."""
        from src.data.loader import ASLDataLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_dataset(tmpdir, class_names=("A", "B"), n_per_class=10)
            loader = ASLDataLoader(data_dir=tmpdir, image_size=16,
                                   color_mode="grayscale")
            (X_tr, _), _, _ = loader.load_dataset()
            self.assertEqual(X_tr.shape[-1], 1)

    def test_class_names_not_available_before_load(self):
        """Accessing class_names before load_dataset raises RuntimeError."""
        from src.data.loader import ASLDataLoader
        loader = ASLDataLoader(data_dir="data/raw")
        with self.assertRaises(RuntimeError):
            _ = loader.class_names


class TestASLAugmentor(unittest.TestCase):
    """Tests for src/data/augmentation.py"""

    def test_keras_generator_no_flip(self):
        """Test that Keras generator has horizontal_flip=False."""
        from src.data.augmentation import ASLAugmentor
        aug = ASLAugmentor()
        gen = aug.build_keras_generator()
        self.assertFalse(gen.horizontal_flip)

    def test_augment_batch_shape(self):
        """Test that augmented batch preserves shape."""
        from src.data.augmentation import ASLAugmentor
        aug = ASLAugmentor()
        batch = np.random.rand(8, 28, 28, 1).astype(np.float32)
        augmented = aug.augment_batch(batch, seed=42)
        self.assertEqual(augmented.shape, batch.shape)

    def test_tf_augmentation_layer_built(self):
        """Test that TF augmentation Sequential layer is created."""
        from src.data.augmentation import ASLAugmentor
        aug = ASLAugmentor()
        layer = aug.build_tf_augmentation_layer()
        self.assertIsNotNone(layer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
