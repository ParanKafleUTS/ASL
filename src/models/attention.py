"""Attention mechanisms for ASL sign detection.

Implements:
- Channel Attention (Squeeze-and-Excitation)
- Spatial Attention
- Multi-Head Self-Attention for skeleton data
- Skeleton-guided spatial attention for images
"""

import logging
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class SqueezeExcitation(layers.Layer):
    """Channel Attention via Squeeze-and-Excitation (SE) block.

    Recalibrates channel-wise feature responses by explicitly modelling
    interdependencies between channels.

    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.

    Args:
        ratio: Reduction ratio for the bottleneck (default 16).
    """

    def __init__(self, ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        n_channels = input_shape[-1]
        self.fc1 = layers.Dense(max(1, n_channels // self.ratio),
                                activation="relu", use_bias=False)
        self.fc2 = layers.Dense(n_channels, activation="sigmoid", use_bias=False)
        self.gap = layers.GlobalAveragePooling2D()
        super().build(input_shape)

    def call(self, inputs):
        """Apply SE channel attention.

        Args:
            inputs: Feature map (batch, H, W, C).

        Returns:
            Recalibrated feature map (batch, H, W, C).
        """
        squeeze = self.gap(inputs)  # (batch, C)
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)
        excitation = tf.reshape(excitation, (-1, 1, 1, inputs.shape[-1]))
        return inputs * excitation

    def get_config(self):
        config = super().get_config()
        config["ratio"] = self.ratio
        return config


class SpatialAttention(layers.Layer):
    """Spatial attention module for highlighting relevant regions.

    Computes attention weights across spatial dimensions using
    channel-wise statistics (mean and max pooling).

    Reference: CBAM (Convolutional Block Attention Module), ECCV 2018.

    Args:
        kernel_size: Convolution kernel size (default 7).
    """

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding="same",
                                  activation="sigmoid", use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        """Apply spatial attention.

        Args:
            inputs: Feature map (batch, H, W, C).

        Returns:
            Attended feature map (batch, H, W, C).
        """
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config["kernel_size"] = self.kernel_size
        return config


class CBAM(layers.Layer):
    """Convolutional Block Attention Module (CBAM).

    Combines channel attention (SE) and spatial attention sequentially.

    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

    Args:
        ratio: Channel reduction ratio for SE block.
        kernel_size: Kernel size for spatial attention.
    """

    def __init__(self, ratio: int = 16, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channel_att = SqueezeExcitation(ratio=self.ratio)
        self.spatial_att = SpatialAttention(kernel_size=self.kernel_size)
        super().build(input_shape)

    def call(self, inputs):
        x = self.channel_att(inputs)
        x = self.spatial_att(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio, "kernel_size": self.kernel_size})
        return config


def build_attention_cnn(input_shape: Tuple[int, int, int] = (28, 28, 1),
                        num_classes: int = 24,
                        dropout_rate: float = 0.5,
                        learning_rate: float = 0.001,
                        attention_type: str = "cbam") -> keras.Model:
    """Build a CNN with attention mechanisms for ASL detection.

    Args:
        input_shape: Input image shape (H, W, C).
        num_classes: Number of output classes.
        dropout_rate: Dropout rate.
        learning_rate: Optimizer learning rate.
        attention_type: Type of attention ('cbam', 'se', 'spatial').

    Returns:
        Compiled Keras model with attention.
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # Apply attention after first block
    if attention_type in ("cbam", "se"):
        x = SqueezeExcitation(ratio=8, name="se_block_1")(x)
    if attention_type in ("cbam", "spatial"):
        x = SpatialAttention(kernel_size=7, name="spatial_att_1")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if attention_type in ("cbam", "se"):
        x = SqueezeExcitation(ratio=8, name="se_block_2")(x)
    if attention_type in ("cbam", "spatial"):
        x = SpatialAttention(kernel_size=5, name="spatial_att_2")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if attention_type in ("cbam", "se"):
        x = SqueezeExcitation(ratio=8, name="se_block_3")(x)
    x = layers.MaxPooling2D(2)(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs,
                        name=f"attention_cnn_{attention_type}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info(f"Built attention CNN ({attention_type}): {model.count_params():,} params")
    return model
