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
    """Tests for src/data/loader.py (using mock data)."""

    def _create_mock_csv(self, tmpdir: str, filename: str,
                          n_samples: int = 100, num_classes: int = 24) -> str:
        """Create a mock Sign Language MNIST CSV file."""
        import pandas as pd
        labels = np.random.randint(0, num_classes, size=n_samples)
        pixels = np.random.randint(0, 256,
                                   size=(n_samples, 28 * 28)).astype(np.int32)
        df = pd.DataFrame(pixels, columns=[f"pixel{i}" for i in range(28 * 28)])
        df.insert(0, "label", labels)
        filepath = os.path.join(tmpdir, filename)
        df.to_csv(filepath, index=False)
        return filepath

    def test_load_csv(self):
        """Test loading images and labels from CSV."""
        from src.data.loader import ASLDataLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._create_mock_csv(tmpdir, "sign_mnist_train.csv")
            loader = ASLDataLoader(data_dir=tmpdir)
            images, labels = loader.load_csv(filepath)
            self.assertEqual(images.shape, (100, 28, 28))
            self.assertEqual(len(labels), 100)
            self.assertTrue(images.max() <= 1.0)
            self.assertTrue(images.min() >= 0.0)

    def test_load_dataset_split(self):
        """Test that load_dataset creates proper train/val/test splits."""
        from src.data.loader import ASLDataLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_csv(tmpdir, "sign_mnist_train.csv", n_samples=200)
            self._create_mock_csv(tmpdir, "sign_mnist_test.csv", n_samples=50)
            loader = ASLDataLoader(data_dir=tmpdir)
            (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = loader.load_dataset(val_split=0.15)
            # Check shapes
            self.assertEqual(X_tr.ndim, 4)  # (N, 28, 28, 1)
            self.assertEqual(X_v.ndim, 4)
            self.assertEqual(X_te.ndim, 4)
            # Check normalization
            self.assertTrue(X_tr.max() <= 1.0)

    def test_class_weights_computed(self):
        """Test that class weights are computed after loading dataset."""
        from src.data.loader import ASLDataLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_csv(tmpdir, "sign_mnist_train.csv", n_samples=200)
            self._create_mock_csv(tmpdir, "sign_mnist_test.csv", n_samples=50)
            loader = ASLDataLoader(data_dir=tmpdir)
            loader.load_dataset()
            self.assertIsNotNone(loader.class_weights)
            self.assertIsInstance(loader.class_weights, dict)

    def test_missing_file_raises(self):
        """Test that FileNotFoundError is raised for missing dataset."""
        from src.data.loader import ASLDataLoader
        loader = ASLDataLoader(data_dir="/nonexistent/path")
        with self.assertRaises(FileNotFoundError):
            loader.load_dataset()

    def test_get_label_name(self):
        """Test label index to ASL letter conversion."""
        from src.data.loader import ASLDataLoader
        loader = ASLDataLoader()
        self.assertEqual(loader.get_label_name(0), "A")
        self.assertEqual(loader.get_label_name(1), "B")
        # Label 9 should be 'K' (skipping J=9 in ASL labels)
        self.assertEqual(loader.get_label_name(9), "K")


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
