"""Transfer learning models for ASL detection: MobileNetV2 and EfficientNetB0.

These models use pre-trained ImageNet weights and are adapted for 
ASL hand sign classification. Input images are upscaled from 28x28 to
the minimum input size required by each architecture.
"""

import logging
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

# Minimum input sizes for transfer learning models
MOBILENET_MIN_SIZE = 96
EFFICIENTNET_MIN_SIZE = 32


def build_mobilenet(input_shape: Tuple[int, int, int] = (96, 96, 1),
                    num_classes: int = 24,
                    dropout_rate: float = 0.3,
                    learning_rate: float = 1e-4,
                    fine_tune_layers: int = 30,
                    freeze_base: bool = True) -> keras.Model:
    """Build a MobileNetV2-based transfer learning model for ASL.

    MobileNetV2 is optimized for mobile and embedded applications while
    maintaining competitive accuracy. The model uses depthwise separable
    convolutions for efficiency.

    Args:
        input_shape: Input image shape (H, W, C). Min 96x96 recommended.
        num_classes: Number of ASL sign classes.
        dropout_rate: Dropout rate for the classification head.
        learning_rate: Initial learning rate for Adam optimizer.
        fine_tune_layers: Number of layers from top to unfreeze for fine-tuning.
        freeze_base: If True, freeze the base model initially.

    Returns:
        Compiled Keras model.
    """
    # MobileNetV2 expects 3 channels
    h, w, c = input_shape
    img_input = keras.Input(shape=input_shape, name="image_input")

    # Convert grayscale to RGB if needed
    if c == 1:
        x = layers.Conv2D(3, 1, padding="same", name="grayscale_to_rgb")(img_input)
    else:
        x = img_input

    # Resize to MobileNetV2 minimum input size
    if h < MOBILENET_MIN_SIZE or w < MOBILENET_MIN_SIZE:
        x = layers.Resizing(MOBILENET_MIN_SIZE, MOBILENET_MIN_SIZE, name="resize")(x)

    # Scale to [0, 255] range that MobileNetV2 preprocessing expects
    x = layers.Rescaling(255.0, name="rescale_0_255")(x)

    # Pre-process for MobileNetV2
    x = layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenet_preprocess"
    )(x)

    # Base model
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )
    base_model.trainable = not freeze_base

    # Unfreeze top layers for fine-tuning
    if fine_tune_layers > 0 and not freeze_base:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(img_input, outputs, name="mobilenet_v2_asl")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info(f"Built MobileNetV2 model: {model.count_params():,} parameters")
    return model


def build_efficientnet(input_shape: Tuple[int, int, int] = (32, 32, 1),
                       num_classes: int = 24,
                       dropout_rate: float = 0.3,
                       learning_rate: float = 1e-4,
                       fine_tune_layers: int = 20,
                       freeze_base: bool = True) -> keras.Model:
    """Build an EfficientNetB0-based transfer learning model for ASL.

    EfficientNetB0 provides state-of-the-art accuracy with efficient scaling.
    It's well-suited for ASL detection due to its compound scaling approach.

    Args:
        input_shape: Input image shape (H, W, C). Min 32x32.
        num_classes: Number of ASL sign classes.
        dropout_rate: Dropout rate for classification head.
        learning_rate: Initial learning rate.
        fine_tune_layers: Top layers to unfreeze for fine-tuning.
        freeze_base: If True, freeze the base model initially.

    Returns:
        Compiled Keras model.
    """
    h, w, c = input_shape
    img_input = keras.Input(shape=input_shape, name="image_input")

    # Convert grayscale to RGB
    if c == 1:
        x = layers.Conv2D(3, 1, padding="same", name="grayscale_to_rgb")(img_input)
    else:
        x = img_input

    # EfficientNet minimum size is 32x32
    if h < EFFICIENTNET_MIN_SIZE or w < EFFICIENTNET_MIN_SIZE:
        x = layers.Resizing(EFFICIENTNET_MIN_SIZE, EFFICIENTNET_MIN_SIZE, name="resize")(x)

    # Scale to [0, 255]
    x = layers.Rescaling(255.0, name="rescale_0_255")(x)

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )
    base_model.trainable = not freeze_base

    if fine_tune_layers > 0 and not freeze_base:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(img_input, outputs, name="efficientnet_b0_asl")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info(f"Built EfficientNetB0 model: {model.count_params():,} parameters")
    return model


def unfreeze_and_finetune(model: keras.Model, num_layers: int,
                          learning_rate: float = 1e-5) -> keras.Model:
    """Unfreeze top layers of a transfer learning model for fine-tuning.

    Call this after initial training with frozen base to fine-tune the
    top layers with a lower learning rate.

    Args:
        model: Previously trained model with frozen base.
        num_layers: Number of layers from the top to unfreeze.
        learning_rate: Reduced learning rate for fine-tuning.

    Returns:
        Recompiled model with unfrozen layers.
    """
    # Find the base model (usually the large sub-model)
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            base_model = layer
            break
    else:
        logger.warning("Could not find base model, unfreezing all layers")
        model.trainable = True
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # Unfreeze top N layers
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    trainable_count = sum(1 for l in model.layers if l.trainable)
    logger.info(f"Fine-tuning: {trainable_count} trainable layers, lr={learning_rate}")
    return model
