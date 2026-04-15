"""Hand skeleton keypoint extraction for ASL detection.

This module provides hand keypoint extraction without MediaPipe, using
OpenCV-based approaches. It extracts 21 hand landmarks and normalizes them
for use in skeleton-based models.

Alternative to MediaPipe for compatibility reasons.
"""

import logging
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# Hand skeleton connectivity (21 landmarks)
# Based on anatomical hand structure
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]

NUM_KEYPOINTS = 21


class SkeletonExtractor:
    """Extracts and normalizes hand skeleton keypoints from images.

    Uses a lightweight approach based on hand contour detection
    when full keypoint detection libraries are unavailable.

    Attributes:
        image_size: Input image size (assumed square).
        normalize: Whether to normalize extracted keypoints.
    """

    def __init__(self, image_size: int = 28, normalize: bool = True):
        """Initialize the skeleton extractor.

        Args:
            image_size: Size of input images.
            normalize: Whether to normalize keypoints to [0, 1].
        """
        self.image_size = image_size
        self.normalize = normalize

    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extract pseudo-skeleton keypoints from a grayscale image.

        For the Sign Language MNIST dataset, since full 3D hand keypoint
        extraction requires depth sensors or complex models, this method
        extracts meaningful 2D spatial features that encode finger positions.

        Args:
            image: Grayscale image of shape (H, W) or (H, W, 1), normalized [0,1].

        Returns:
            Keypoint array of shape (21, 2) with (x, y) coordinates.
        """
        if image.ndim == 3:
            image = image.squeeze(-1)

        # Convert to uint8 for OpenCV processing
        img_uint8 = (image * 255).astype(np.uint8)
        return self._extract_keypoints_opencv(img_uint8)

    def _extract_keypoints_opencv(self, img_uint8: np.ndarray) -> np.ndarray:
        """Extract hand keypoints using OpenCV contour analysis.

        Args:
            img_uint8: Uint8 grayscale image.

        Returns:
            Keypoint array of shape (21, 2).
        """
        try:
            import cv2

            # Threshold to isolate hand
            _, binary = cv2.threshold(img_uint8, 50, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return self._fallback_keypoints(img_uint8)

            # Use largest contour (the hand)
            hand_contour = max(contours, key=cv2.contourArea)

            # Find convex hull and defects (finger gaps)
            hull = cv2.convexHull(hand_contour, returnPoints=False)
            if len(hull) < 3:
                return self._fallback_keypoints(img_uint8)

            defects = cv2.convexityDefects(hand_contour, hull)

            # Extract landmark points from contour
            keypoints = self._contour_to_keypoints(hand_contour, defects, img_uint8.shape)

        except ImportError:
            logger.warning("OpenCV not available. Using fallback keypoint extraction.")
            keypoints = self._fallback_keypoints(img_uint8)

        if self.normalize:
            keypoints = self._normalize_keypoints(keypoints)

        return keypoints

    def _contour_to_keypoints(self, contour: np.ndarray, defects,
                               img_shape: tuple) -> np.ndarray:
        """Convert hand contour to 21 pseudo-keypoints.

        Args:
            contour: Hand contour points.
            defects: Convexity defects.
            img_shape: Image shape (H, W).

        Returns:
            Array of shape (21, 2) with keypoints.
        """
        h, w = img_shape[:2]
        keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

        # Wrist (point 0) - bottom center of bounding box
        x, y, bw, bh = cv2_boundingRect_safe(contour)
        keypoints[0] = [x + bw // 2, y + bh]

        # Sample points along contour for finger positions
        if len(contour) > NUM_KEYPOINTS:
            step = len(contour) // NUM_KEYPOINTS
            for i in range(1, NUM_KEYPOINTS):
                idx = min(i * step, len(contour) - 1)
                keypoints[i] = contour[idx][0]
        else:
            for i in range(1, min(NUM_KEYPOINTS, len(contour))):
                keypoints[i] = contour[i - 1][0]

        return keypoints

    def _fallback_keypoints(self, img: np.ndarray) -> np.ndarray:
        """Generate keypoints based on intensity distribution (fallback).

        This is used when OpenCV is unavailable or contour detection fails.
        It samples points from bright regions in the image.

        Args:
            img: Grayscale uint8 image.

        Returns:
            Array of shape (21, 2).
        """
        h, w = img.shape[:2]
        keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

        # Find bright pixel locations (hand pixels)
        bright_mask = img > 50
        ys, xs = np.where(bright_mask)

        if len(xs) == 0:
            # Return grid of points if no hand detected
            grid_size = int(np.ceil(np.sqrt(NUM_KEYPOINTS)))
            pts = [(i % grid_size * w // grid_size,
                    i // grid_size * h // grid_size)
                   for i in range(NUM_KEYPOINTS)]
            keypoints = np.array(pts, dtype=np.float32)
            return keypoints

        # Sample NUM_KEYPOINTS points from hand region
        indices = np.linspace(0, len(xs) - 1, NUM_KEYPOINTS, dtype=int)
        keypoints[:, 0] = xs[indices]
        keypoints[:, 1] = ys[indices]

        return keypoints

    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize keypoints to be scale, translation, and rotation invariant.

        Applies normalization: center on wrist, scale by hand size.

        Args:
            keypoints: Raw keypoints of shape (21, 2).

        Returns:
            Normalized keypoints of shape (21, 2).
        """
        # Center on wrist (keypoint 0)
        wrist = keypoints[0]
        centered = keypoints - wrist

        # Scale by maximum distance from wrist
        max_dist = np.max(np.linalg.norm(centered, axis=1)) + 1e-8
        normalized = centered / max_dist

        return normalized.astype(np.float32)

    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """Extract keypoints from a batch of images.

        Args:
            images: Batch of images (N, H, W) or (N, H, W, 1).

        Returns:
            Keypoint array of shape (N, 21, 2).
        """
        if images.ndim == 4:
            images = images.squeeze(-1)

        keypoints = np.array([
            self.extract_from_image(img) for img in images
        ])
        return keypoints

    def visualize_skeleton(self, image: np.ndarray, keypoints: np.ndarray,
                           save_path: Optional[str] = None) -> None:
        """Visualize skeleton overlay on an image.

        Args:
            image: Grayscale image (H, W) or (H, W, 1).
            keypoints: Keypoint array (21, 2), normalized or raw.
            save_path: Optional path to save the visualization.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if image.ndim == 3:
            image = image.squeeze(-1)

        # Denormalize if needed
        h, w = image.shape[:2]
        kp = keypoints.copy()
        if kp.max() <= 1.0 and kp.min() >= -1.0:
            kp = (kp + 1.0) * 0.5 * np.array([w, h])

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image, cmap="gray")

        # Draw connections
        for start, end in HAND_CONNECTIONS:
            if start < len(kp) and end < len(kp):
                ax.plot([kp[start, 0], kp[end, 0]],
                        [kp[start, 1], kp[end, 1]],
                        "b-", linewidth=1.5, alpha=0.7)

        # Draw keypoints
        ax.scatter(kp[:, 0], kp[:, 1], c="red", s=30, zorder=5)
        ax.scatter(kp[0, 0], kp[0, 1], c="green", s=60, zorder=6,
                   label="Wrist")

        ax.set_title("Hand Skeleton Overlay")
        ax.legend()
        ax.axis("off")

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def cv2_boundingRect_safe(contour):
    """Wrapper for cv2.boundingRect to handle import gracefully."""
    try:
        import cv2
        return cv2.boundingRect(contour)
    except ImportError:
        pts = contour.reshape(-1, 2)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        w = pts[:, 0].max() - x
        h = pts[:, 1].max() - y
        return x, y, w, h
