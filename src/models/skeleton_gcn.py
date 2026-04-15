"""Graph Convolutional Network (GCN) on hand skeleton for ASL detection.

This is the novel contribution of the thesis: applying GCN directly to
hand skeleton graphs for ASL sign classification.

The hand skeleton is modeled as a graph where:
- Nodes = 21 hand keypoints (x, y coordinates)
- Edges = anatomical connections between keypoints

Architecture: Stacked GCN layers with residual connections and pooling.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

# Adjacency matrix for hand skeleton (21 nodes)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


def build_adjacency_matrix(n_nodes: int = 21,
                            edges: List[Tuple[int, int]] = None,
                            normalized: bool = True) -> np.ndarray:
    """Build the hand skeleton adjacency matrix.

    Args:
        n_nodes: Number of skeleton nodes (default 21 for hand).
        edges: List of (start, end) edge tuples.
        normalized: If True, normalize by degree (D^-1/2 A D^-1/2).

    Returns:
        Adjacency matrix of shape (n_nodes, n_nodes).
    """
    if edges is None:
        edges = HAND_EDGES

    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0  # Undirected graph

    # Self-connections
    A += np.eye(n_nodes, dtype=np.float32)

    if normalized:
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        D = np.diag(A.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
        A = D_inv_sqrt @ A @ D_inv_sqrt

    return A


class GraphConvLayer(layers.Layer):
    """Single Graph Convolutional Layer.

    Implements: H' = sigma(A H W + b)
    where A is the (normalized) adjacency matrix.

    Args:
        units: Number of output features.
        activation: Activation function name.
        use_bias: Whether to add bias.
    """

    def __init__(self, units: int, activation: str = "relu",
                 use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        n_features = input_shape[-1]
        self.W = self.add_weight(
            name="W", shape=(n_features, self.units),
            initializer="glorot_uniform", trainable=True
        )
        if self.use_bias:
            self.b = self.add_weight(
                name="b", shape=(self.units,),
                initializer="zeros", trainable=True
            )
        super().build(input_shape)

    def call(self, inputs, adjacency):
        """Forward pass.

        Args:
            inputs: Node features (batch_size, n_nodes, n_features).
            adjacency: Adjacency matrix (n_nodes, n_nodes).

        Returns:
            Updated node features (batch_size, n_nodes, units).
        """
        # H W
        support = tf.matmul(inputs, self.W)
        # A H W
        output = tf.matmul(adjacency, support)

        if self.use_bias:
            output += self.b

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
        })
        return config


class SkeletonGCN(keras.Model):
    """Graph Convolutional Network for skeleton-based ASL classification.

    Takes hand skeleton keypoints as input and classifies ASL signs using
    graph convolution over the hand's anatomical connectivity.

    Attributes:
        gcn_units: Hidden units in each GCN layer.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability.
    """

    def __init__(self, gcn_units: List[int] = None,
                 num_classes: int = 24,
                 dropout_rate: float = 0.4,
                 n_nodes: int = 21,
                 **kwargs):
        """Initialize the Skeleton GCN.

        Args:
            gcn_units: List of hidden units for each GCN layer.
            num_classes: Number of output classes.
            dropout_rate: Dropout rate for regularization.
            n_nodes: Number of skeleton nodes.
        """
        super().__init__(**kwargs)
        if gcn_units is None:
            gcn_units = [64, 128, 64]

        self.gcn_units = gcn_units
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.n_nodes = n_nodes

        # Build GCN layers
        self.gcn_layers = []
        self.bn_layers = []
        self.dropout_layers = []

        for i, units in enumerate(gcn_units):
            self.gcn_layers.append(
                GraphConvLayer(units, activation="relu", name=f"gcn_{i+1}")
            )
            self.bn_layers.append(
                layers.BatchNormalization(name=f"gcn_bn_{i+1}")
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f"gcn_dropout_{i+1}")
            )

        # Classification head
        self.flatten = layers.Flatten(name="flatten")
        self.dense1 = layers.Dense(128, activation="relu", name="dense_1")
        self.bn_dense = layers.BatchNormalization(name="bn_dense")
        self.dropout_final = layers.Dropout(dropout_rate, name="dropout_final")
        self.classifier = layers.Dense(num_classes, activation="softmax",
                                       name="predictions")

        # Pre-compute adjacency matrix (constant)
        self.adjacency = tf.constant(
            build_adjacency_matrix(n_nodes, normalized=True),
            dtype=tf.float32
        )

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: Skeleton keypoints (batch_size, n_nodes, 2).
            training: Whether in training mode.

        Returns:
            Class probabilities (batch_size, num_classes).
        """
        x = inputs  # (batch_size, 21, 2)

        for gcn, bn, drop in zip(self.gcn_layers, self.bn_layers, self.dropout_layers):
            x = gcn(x, self.adjacency)
            x = bn(x, training=training)
            x = drop(x, training=training)

        # Global pooling over nodes
        x = tf.reduce_mean(x, axis=1)  # (batch_size, gcn_units[-1])

        x = self.dense1(x)
        x = self.bn_dense(x, training=training)
        x = self.dropout_final(x, training=training)
        return self.classifier(x)

    def get_config(self):
        return {
            "gcn_units": self.gcn_units,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "n_nodes": self.n_nodes,
        }


def build_skeleton_gcn(n_nodes: int = 21,
                       n_features: int = 2,
                       num_classes: int = 24,
                       gcn_units: List[int] = None,
                       dropout_rate: float = 0.4,
                       learning_rate: float = 0.001) -> keras.Model:
    """Build and compile a Skeleton GCN model using functional API.

    Args:
        n_nodes: Number of skeleton keypoints (default 21).
        n_features: Features per node (default 2 for x, y).
        num_classes: Number of output classes.
        gcn_units: Hidden units per GCN layer.
        dropout_rate: Dropout rate.
        learning_rate: Optimizer learning rate.

    Returns:
        Compiled Keras model accepting skeleton input of shape (n_nodes, n_features).
    """
    if gcn_units is None:
        gcn_units = [64, 128, 64]

    model = SkeletonGCN(
        gcn_units=gcn_units,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        n_nodes=n_nodes,
        name="skeleton_gcn"
    )

    # Build by running a dummy forward pass
    dummy_input = tf.zeros((1, n_nodes, n_features))
    _ = model(dummy_input)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info(f"Built SkeletonGCN: {model.count_params():,} parameters")
    return model
