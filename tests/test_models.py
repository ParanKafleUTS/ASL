"""Unit tests for model architectures."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestCustomCNN(unittest.TestCase):
    """Tests for src/models/custom_cnn.py"""

    def test_build_default(self):
        """Test building default custom CNN."""
        from src.models.custom_cnn import build_custom_cnn
        model = build_custom_cnn()
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape, (None, 24))

    def test_forward_pass(self):
        """Test a forward pass through the custom CNN."""
        from src.models.custom_cnn import build_custom_cnn
        model = build_custom_cnn(input_shape=(28, 28, 1), num_classes=24)
        x = np.random.rand(4, 28, 28, 1).astype(np.float32)
        preds = model.predict(x, verbose=0)
        self.assertEqual(preds.shape, (4, 24))
        # Check probabilities sum to 1
        np.testing.assert_allclose(preds.sum(axis=1),
                                   np.ones(4), atol=1e-5)

    def test_custom_filters(self):
        """Test CNN with custom filter configuration."""
        from src.models.custom_cnn import build_custom_cnn
        model = build_custom_cnn(filters=[16, 32], dense_units=[64])
        self.assertIsNotNone(model)

    def test_parameter_count_reasonable(self):
        """Test that model has a reasonable number of parameters."""
        from src.models.custom_cnn import build_custom_cnn
        model = build_custom_cnn()
        params = model.count_params()
        # Should be between 10k and 5M parameters
        self.assertGreater(params, 10_000)
        self.assertLess(params, 5_000_000)


class TestSkeletonGCN(unittest.TestCase):
    """Tests for src/models/skeleton_gcn.py"""

    def test_build_gcn(self):
        """Test building the skeleton GCN."""
        from src.models.skeleton_gcn import build_skeleton_gcn
        model = build_skeleton_gcn(n_nodes=21, n_features=2, num_classes=24)
        self.assertIsNotNone(model)

    def test_adjacency_matrix(self):
        """Test adjacency matrix construction."""
        from src.models.skeleton_gcn import build_adjacency_matrix
        A = build_adjacency_matrix(n_nodes=21)
        self.assertEqual(A.shape, (21, 21))
        # Check symmetry
        np.testing.assert_allclose(A, A.T, atol=1e-6)
        # Check self-connections (diagonal should be non-zero before normalization)

    def test_gcn_forward_pass(self):
        """Test GCN forward pass with skeleton input."""
        from src.models.skeleton_gcn import build_skeleton_gcn
        model = build_skeleton_gcn(n_nodes=21, n_features=2, num_classes=24)
        x = np.random.rand(4, 21, 2).astype(np.float32)
        preds = model.predict(x, verbose=0)
        self.assertEqual(preds.shape, (4, 24))
        np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-5)


class TestAttentionModels(unittest.TestCase):
    """Tests for src/models/attention.py"""

    def test_build_attention_cnn_cbam(self):
        """Test building CBAM attention CNN."""
        from src.models.attention import build_attention_cnn
        model = build_attention_cnn(attention_type="cbam")
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape, (None, 24))

    def test_build_attention_cnn_se(self):
        """Test building SE-block attention CNN."""
        from src.models.attention import build_attention_cnn
        model = build_attention_cnn(attention_type="se")
        self.assertIsNotNone(model)

    def test_se_block_forward(self):
        """Test SE block forward pass."""
        from src.models.attention import SqueezeExcitation
        import tensorflow as tf
        se = SqueezeExcitation(ratio=4)
        x = tf.random.uniform((2, 7, 7, 32))
        out = se(x)
        self.assertEqual(out.shape, x.shape)

    def test_spatial_attention_forward(self):
        """Test spatial attention forward pass."""
        from src.models.attention import SpatialAttention
        import tensorflow as tf
        att = SpatialAttention(kernel_size=3)
        x = tf.random.uniform((2, 7, 7, 32))
        out = att(x)
        self.assertEqual(out.shape, x.shape)


class TestSkeletonExtractor(unittest.TestCase):
    """Tests for src/models/skeleton_extraction.py"""

    def test_extract_from_image_shape(self):
        """Test that extraction returns correct keypoint shape."""
        from src.models.skeleton_extraction import SkeletonExtractor
        extractor = SkeletonExtractor()
        image = np.random.rand(28, 28).astype(np.float32)
        keypoints = extractor.extract_from_image(image)
        self.assertEqual(keypoints.shape, (21, 2))

    def test_extract_batch_shape(self):
        """Test batch extraction shape."""
        from src.models.skeleton_extraction import SkeletonExtractor
        extractor = SkeletonExtractor()
        images = np.random.rand(8, 28, 28, 1).astype(np.float32)
        keypoints = extractor.extract_batch(images)
        self.assertEqual(keypoints.shape, (8, 21, 2))

    def test_normalized_keypoints_range(self):
        """Test that normalized keypoints have bounded range."""
        from src.models.skeleton_extraction import SkeletonExtractor
        extractor = SkeletonExtractor(normalize=True)
        image = np.random.rand(28, 28).astype(np.float32)
        keypoints = extractor.extract_from_image(image)
        # Wrist at (0, 0) after normalization
        np.testing.assert_allclose(keypoints[0], [0, 0], atol=1e-5)


class TestEnsemble(unittest.TestCase):
    """Tests for src/models/ensemble.py"""

    def _make_dummy_model(self, num_classes: int = 24):
        """Create a simple dummy Keras model for testing."""
        import tensorflow as tf
        from tensorflow import keras
        inputs = keras.Input(shape=(28, 28, 1))
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs, x)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def test_ensemble_creation(self):
        """Test ensemble creation with multiple models."""
        from src.models.ensemble import EnsembleModel
        models = [self._make_dummy_model() for _ in range(3)]
        ensemble = EnsembleModel(models, strategy="weighted_average")
        self.assertEqual(len(ensemble.models), 3)

    def test_ensemble_requires_min_models(self):
        """Test that single model raises error."""
        from src.models.ensemble import EnsembleModel
        models = [self._make_dummy_model()]
        with self.assertRaises(ValueError):
            EnsembleModel(models)

    def test_weighted_average_prediction(self):
        """Test weighted average ensemble prediction."""
        from src.models.ensemble import EnsembleModel
        models = [self._make_dummy_model() for _ in range(2)]
        ensemble = EnsembleModel(models, strategy="weighted_average")
        X = np.random.rand(10, 28, 28, 1).astype(np.float32)
        probs = ensemble.predict_proba(X)
        self.assertEqual(probs.shape, (10, 24))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-4)

    def test_voting_prediction(self):
        """Test voting ensemble prediction."""
        from src.models.ensemble import EnsembleModel
        models = [self._make_dummy_model() for _ in range(3)]
        ensemble = EnsembleModel(models, strategy="voting")
        X = np.random.rand(10, 28, 28, 1).astype(np.float32)
        probs = ensemble.predict_proba(X)
        self.assertEqual(probs.shape, (10, 24))

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        from src.models.ensemble import EnsembleModel
        models = [self._make_dummy_model() for _ in range(2)]
        ensemble = EnsembleModel(models, weights=[2.0, 3.0])
        self.assertAlmostEqual(sum(ensemble.weights), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
