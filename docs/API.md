# API Documentation

## Package: `src`

---

## `src.data.loader`

### `ASLDataLoader`

Loads and preprocesses the ASL Alphabet image dataset (`grassknoted/asl-alphabet`).
Images are read from a directory hierarchy where each sub-folder is a class label.
The loader auto-detects flat, single-nested, and doubly-nested layouts
(e.g. `asl_alphabet_train/asl_alphabet_train/{class}/`).

```python
from src.data.loader import ASLDataLoader

loader = ASLDataLoader(data_dir='data/raw', image_size=64, color_mode='rgb')
(X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.load_dataset(val_split=0.15)
```

**Parameters:**
- `data_dir` (str): Root directory populated by `data/download_dataset.py`.
- `image_size` (int): Images are resized to `(image_size, image_size)`. Default `64`.
- `color_mode` (str): `"rgb"` (3 channels, default) or `"grayscale"` (1 channel).

**Properties:**
- `class_names` → `List[str]` — Sorted list of class names (available after `load_dataset`).
- `num_classes` → `int` — Number of detected classes.
- `class_weights` → `dict` — Balanced weights for imbalanced classes.

**Methods:**
- `load_dataset(val_split)` → `((X_train, y_train), (X_val, y_val), (X_test, y_test))`
- `get_label_name(label_idx)` → `str`
- `visualize_samples(X, y, n_samples, save_path)` — Grid of sample images.
- `visualize_class_distribution(y, title, save_path)` — Bar chart of class counts.

**Dataset classes (29 total):**
`A B C D E F G H I J K L M N O P Q R S T U V W X Y Z del nothing space`

---

## `src.data.augmentation`

### `ASLAugmentation`

Real-time data augmentation pipeline.

```python
from src.data.augmentation import ASLAugmentation

aug = ASLAugmentation(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
datagen = aug.get_train_generator()
datagen.fit(X_train)
```

**Methods:**
- `get_train_generator()` → `ImageDataGenerator`
- `get_val_generator()` → `ImageDataGenerator` (no augmentation)
- `visualize_augmentations(X, y, n_samples)` — Shows augmented samples.

---

## `src.models.custom_cnn`

### `build_custom_cnn(input_shape, num_classes, dropout_rate)`

Builds the baseline 3-layer CNN.

```python
from src.models.custom_cnn import build_custom_cnn

model = build_custom_cnn(input_shape=(28, 28, 1), num_classes=24)
model.summary()
```

**Returns:** `tf.keras.Model`

---

## `src.models.transfer_learning`

### `build_mobilenet(input_shape, num_classes, fine_tune_at)`
### `build_efficientnet(input_shape, num_classes, fine_tune_at)`

```python
from src.models.transfer_learning import build_mobilenet, build_efficientnet

model = build_mobilenet(input_shape=(96, 96, 3), num_classes=24, fine_tune_at=100)
```

**Parameters:**
- `fine_tune_at` (int): Layer index from which to unfreeze for fine-tuning. `None` = feature extraction only.

---

## `src.models.skeleton_gcn`

### `build_skeleton_gcn(num_nodes, num_features, num_classes)`

Graph Convolutional Network operating on hand skeleton keypoints.

```python
from src.models.skeleton_gcn import build_skeleton_gcn

model = build_skeleton_gcn(num_nodes=21, num_features=2, num_classes=24)
```

---

## `src.models.attention`

### `build_attention_cnn(input_shape, num_classes)`

CNN with channel and spatial attention (SE-Net style).

```python
from src.models.attention import build_attention_cnn

model = build_attention_cnn(input_shape=(28, 28, 1), num_classes=24)
```

---

## `src.models.ensemble`

### `EnsembleModel`

Combines predictions from multiple models.

```python
from src.models.ensemble import EnsembleModel

ensemble = EnsembleModel(models=[cnn, mobilenet, gcn], weights=[0.3, 0.4, 0.3])
preds = ensemble.predict(X_test)
```

**Methods:**
- `predict(X)` → `np.ndarray` — Weighted average probabilities.
- `evaluate(X, y)` → `dict` — Accuracy and F1 metrics.

---

## `src.training.trainer`

### `Trainer`

Unified training interface for all models.

```python
from src.training.trainer import Trainer

trainer = Trainer(model, config_path='config.yaml')
history = trainer.train(X_train, y_train, X_val, y_val)
trainer.save_model('results/models/my_model.h5')
```

**Methods:**
- `train(X_train, y_train, X_val, y_val)` → `History`
- `save_model(path)` — Save best model weights.
- `load_model(path)` — Load saved weights.
- `plot_history()` — Plot loss/accuracy curves.

---

## `src.evaluation.metrics`

### `ModelEvaluator`

```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator(model, class_names=loader.class_names)
report = evaluator.full_report(X_test, y_test)
# report contains: accuracy, precision, recall, f1, confusion_matrix,
#                  bootstrap_ci, roc_auc, per_class_metrics
```

**Methods:**
- `full_report(X, y)` → `dict`
- `bootstrap_confidence_interval(X, y, n_bootstrap)` → `(lower, upper)`
- `statistical_significance_test(model_a_preds, model_b_preds, y_true)` → `p_value`

---

## `src.evaluation.visualizations`

### `ResultVisualizer`

```python
from src.evaluation.visualizations import ResultVisualizer

viz = ResultVisualizer(output_dir='results/plots')
viz.plot_training_history(history)
viz.plot_confusion_matrix(y_true, y_pred, class_names)
viz.plot_model_comparison(results_dict)
viz.plot_per_class_f1(report)
viz.plot_roc_curves(models_dict, X_test, y_test)
```

---

## `src.utils.config`

### `load_config(path)`

```python
from src.utils.config import load_config

cfg = load_config('config.yaml')
print(cfg['training']['batch_size'])
```

---

## `src.utils.logger`

### `get_logger(name, log_file)`

```python
from src.utils.logger import get_logger

logger = get_logger('trainer', log_file='results/logs/training.log')
logger.info('Training started')
```
