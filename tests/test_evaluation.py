"""Unit tests for evaluation metrics and analysis modules."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestASLEvaluator(unittest.TestCase):
    """Tests for src/evaluation/metrics.py"""

    def _make_predictions(self, n: int = 100, num_classes: int = 24, seed: int = 42):
        """Generate mock predictions for testing."""
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, num_classes, size=n)
        # Make some predictions that are often correct
        y_pred = y_true.copy()
        noise_idx = rng.choice(n, size=n // 4, replace=False)
        y_pred[noise_idx] = rng.integers(0, num_classes, size=len(noise_idx))
        y_proba = rng.dirichlet(np.ones(num_classes), size=n).astype(np.float32)
        return y_true, y_pred, y_proba

    def test_compute_metrics_keys(self):
        """Test that compute_metrics returns all expected keys."""
        from src.evaluation.metrics import ASLEvaluator
        evaluator = ASLEvaluator()
        y_true, y_pred, _ = self._make_predictions()
        metrics = evaluator.compute_metrics(y_true, y_pred)

        expected_keys = ["accuracy", "f1_macro", "f1_weighted",
                         "precision_macro", "recall_macro", "per_class"]
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_accuracy_correct(self):
        """Test that accuracy computation is correct."""
        from src.evaluation.metrics import ASLEvaluator
        evaluator = ASLEvaluator()
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 0, 3])  # 3/4 correct
        metrics = evaluator.compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=5)

    def test_perfect_predictions(self):
        """Test metrics when predictions are perfect."""
        from src.evaluation.metrics import ASLEvaluator
        evaluator = ASLEvaluator()
        y = np.array([0, 1, 2, 3, 4])
        metrics = evaluator.compute_metrics(y, y)
        self.assertAlmostEqual(metrics["accuracy"], 1.0, places=5)
        self.assertAlmostEqual(metrics["f1_macro"], 1.0, places=5)

    def test_confusion_matrix_shape(self):
        """Test confusion matrix shape."""
        from src.evaluation.metrics import ASLEvaluator
        evaluator = ASLEvaluator(num_classes=5)
        y_true = np.array([0, 1, 2, 3, 4, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 4, 1, 0])
        metrics = evaluator.compute_metrics(y_true, y_pred)
        cm = np.array(metrics["confusion_matrix"])
        self.assertEqual(cm.shape, (5, 5))

    def test_bootstrap_ci_returns_triple(self):
        """Test that bootstrap CI returns (estimate, lower, upper)."""
        from src.evaluation.metrics import ASLEvaluator
        evaluator = ASLEvaluator(bootstrap_samples=100)
        y_true, y_pred, _ = self._make_predictions(n=50)
        result = evaluator.bootstrap_confidence_interval(y_true, y_pred)
        self.assertEqual(len(result), 3)
        estimate, lower, upper = result
        self.assertLessEqual(lower, estimate)
        self.assertLessEqual(estimate, upper)

    def test_mcnemar_test_identical_models(self):
        """Test McNemar's test when models have identical predictions."""
        from src.evaluation.metrics import ASLEvaluator
        evaluator = ASLEvaluator()
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        result = evaluator.mcnemar_test(y_true, y_pred, y_pred)
        # Identical models: no significant difference
        self.assertFalse(result["significant"])
        self.assertEqual(result["b"], 0)
        self.assertEqual(result["c"], 0)

    def test_roc_auc_with_probabilities(self):
        """Test ROC-AUC computation with probability predictions."""
        from src.evaluation.metrics import ASLEvaluator
        evaluator = ASLEvaluator(num_classes=5)
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 5, size=100)
        y_pred = y_true.copy()
        y_proba = rng.dirichlet(np.ones(5), size=100).astype(np.float32)
        metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
        self.assertIn("roc_auc_macro", metrics)
        self.assertGreaterEqual(metrics["roc_auc_macro"], 0.0)
        self.assertLessEqual(metrics["roc_auc_macro"], 1.0)

    def test_save_metrics(self):
        """Test that metrics can be saved to JSON."""
        import tempfile
        import json
        from src.evaluation.metrics import ASLEvaluator
        evaluator = ASLEvaluator()
        y_true, y_pred, _ = self._make_predictions()
        metrics = evaluator.compute_metrics(y_true, y_pred)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.json")
            evaluator.save_metrics(metrics, path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                loaded = json.load(f)
            self.assertIn("accuracy", loaded)


class TestRobustnessTester(unittest.TestCase):
    """Tests for src/evaluation/robustness_testing.py"""

    def test_occlusion_shape_preserved(self):
        """Test that occlusion preserves image shape."""
        from src.evaluation.robustness_testing import RobustnessTester
        tester = RobustnessTester()
        images = np.random.rand(4, 28, 28, 1).astype(np.float32)
        result = tester._add_occlusion(images, 0.25)
        self.assertEqual(result.shape, images.shape)

    def test_noise_shape_preserved(self):
        """Test that noise addition preserves image shape."""
        from src.evaluation.robustness_testing import RobustnessTester
        tester = RobustnessTester()
        images = np.random.rand(4, 28, 28, 1).astype(np.float32)
        result = tester._add_noise(images, std=0.05)
        self.assertEqual(result.shape, images.shape)

    def test_low_light_darkens(self):
        """Test that low-light transformation darkens images."""
        from src.evaluation.robustness_testing import RobustnessTester
        tester = RobustnessTester()
        images = np.ones((4, 28, 28, 1), dtype=np.float32)
        result = tester._low_light(images, factor=0.3)
        np.testing.assert_allclose(result, np.ones_like(images) * 0.3, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
