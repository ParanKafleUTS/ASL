"""Robustness testing for ASL detection models.

Tests model performance under various challenging conditions:
- Rotated images
- Occluded hands (masked regions)
- Low-light / low contrast
- Gaussian noise
- Blurring
"""

import logging
from typing import Dict, List, Optional, Callable

import numpy as np

from .metrics import ASLEvaluator

logger = logging.getLogger(__name__)


class RobustnessTester:
    """Test ASL model robustness under various perturbations.

    Attributes:
        evaluator: ASLEvaluator for metric computation.
        results: Dictionary storing robustness test results.
    """

    def __init__(self, num_classes: int = 24):
        """Initialize the robustness tester.

        Args:
            num_classes: Number of ASL classes.
        """
        self.evaluator = ASLEvaluator(num_classes=num_classes)
        self.results: Dict[str, Dict] = {}

    def test_all(self, model, X_test: np.ndarray,
                 y_test: np.ndarray) -> Dict[str, Dict]:
        """Run all robustness tests.

        Args:
            model: Trained Keras model.
            X_test: Test images of shape (N, H, W, 1).
            y_test: True labels.

        Returns:
            Dictionary of test_name -> metrics.
        """
        tests = {
            "clean": lambda x: x,
            "rotation_15deg": lambda x: self._rotate_images(x, 15),
            "rotation_30deg": lambda x: self._rotate_images(x, 30),
            "occlusion_25pct": lambda x: self._add_occlusion(x, 0.25),
            "occlusion_50pct": lambda x: self._add_occlusion(x, 0.50),
            "low_light": lambda x: self._low_light(x, factor=0.3),
            "gaussian_noise_low": lambda x: self._add_noise(x, std=0.05),
            "gaussian_noise_high": lambda x: self._add_noise(x, std=0.15),
            "blur": lambda x: self._blur_images(x),
        }

        for test_name, transform_fn in tests.items():
            logger.info(f"Running robustness test: {test_name}")
            X_perturbed = transform_fn(X_test.copy())
            X_perturbed = np.clip(X_perturbed, 0.0, 1.0)

            y_proba = model.predict(X_perturbed, batch_size=64, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)

            metrics = self.evaluator.compute_metrics(y_test, y_pred, y_proba)
            self.results[test_name] = {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            }
            logger.info(
                f"  {test_name}: acc={metrics['accuracy']:.4f}"
            )

        return self.results

    def _rotate_images(self, images: np.ndarray, angle: float) -> np.ndarray:
        """Rotate all images by a fixed angle.

        Args:
            images: Image array (N, H, W, 1).
            angle: Rotation angle in degrees.

        Returns:
            Rotated image array.
        """
        try:
            import cv2
            rotated = np.zeros_like(images)
            for i, img in enumerate(images):
                img_2d = img.squeeze(-1)
                h, w = img_2d.shape
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                rot = cv2.warpAffine(img_2d, M, (w, h),
                                     borderMode=cv2.BORDER_REFLECT)
                rotated[i] = rot[:, :, np.newaxis]
            return rotated
        except ImportError:
            logger.warning("OpenCV not available, skipping rotation")
            return images

    def _add_occlusion(self, images: np.ndarray,
                       fraction: float) -> np.ndarray:
        """Add random rectangular occlusion patches.

        Args:
            images: Image array (N, H, W, 1).
            fraction: Fraction of image area to occlude.

        Returns:
            Images with occlusion applied.
        """
        occluded = images.copy()
        _, h, w, _ = images.shape
        patch_h = int(h * np.sqrt(fraction))
        patch_w = int(w * np.sqrt(fraction))

        rng = np.random.default_rng(42)
        for i in range(len(occluded)):
            y0 = rng.integers(0, max(1, h - patch_h))
            x0 = rng.integers(0, max(1, w - patch_w))
            occluded[i, y0:y0 + patch_h, x0:x0 + patch_w, :] = 0.0

        return occluded

    def _low_light(self, images: np.ndarray, factor: float = 0.3) -> np.ndarray:
        """Simulate low-light conditions by darkening images.

        Args:
            images: Image array.
            factor: Brightness multiplier (< 1.0 for darkening).

        Returns:
            Darkened image array.
        """
        return images * factor

    def _add_noise(self, images: np.ndarray, std: float = 0.05) -> np.ndarray:
        """Add Gaussian noise to images.

        Args:
            images: Image array.
            std: Standard deviation of Gaussian noise.

        Returns:
            Noisy image array.
        """
        rng = np.random.default_rng(42)
        noise = rng.normal(0, std, images.shape).astype(np.float32)
        return images + noise

    def _blur_images(self, images: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply Gaussian blur to images.

        Args:
            images: Image array (N, H, W, 1).
            kernel_size: Blur kernel size.

        Returns:
            Blurred image array.
        """
        try:
            import cv2
            blurred = np.zeros_like(images)
            for i, img in enumerate(images):
                img_2d = img.squeeze(-1)
                b = cv2.GaussianBlur(img_2d, (kernel_size, kernel_size), 0)
                blurred[i] = b[:, :, np.newaxis]
            return blurred
        except ImportError:
            return images

    def print_summary(self, baseline_accuracy: Optional[float] = None) -> None:
        """Print robustness test summary.

        Args:
            baseline_accuracy: Clean accuracy for comparison.
        """
        print("\n=== Robustness Test Results ===")
        print(f"{'Test':<30} {'Accuracy':>10} {'F1 Macro':>10} {'Drop':>8}")
        print("-" * 62)

        clean_acc = self.results.get("clean", {}).get("accuracy",
                                                       baseline_accuracy or 0)
        for name, result in self.results.items():
            drop = clean_acc - result["accuracy"]
            print(f"{name:<30} {result['accuracy']:>10.4f} "
                  f"{result['f1_macro']:>10.4f} {drop:>8.4f}")
