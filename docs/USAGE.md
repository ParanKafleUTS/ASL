# Usage Guide

## Quick Start

After completing [installation](INSTALLATION.md), you can train and evaluate models with a few commands.

---

## 1. Data Preparation

### Download Dataset

```bash
python data/download_dataset.py
```

Or directly in Python:

```python
import kagglehub
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Path to dataset files:", path)
```

### Explore the Data

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Or programmatically:

```python
from src.data.loader import ASLDataLoader

loader = ASLDataLoader(data_dir='data/raw')
(X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.load_dataset()
loader.visualize_class_distribution(y_train)
loader.visualize_samples(X_train, y_train)
```

---

## 2. Training Models

### Train All Models

```bash
python -m src.training.trainer --config config.yaml --model all
```

### Train a Specific Model

```bash
# Custom CNN baseline
python -m src.training.trainer --config config.yaml --model custom_cnn

# MobileNetV2 transfer learning
python -m src.training.trainer --config config.yaml --model mobilenet

# EfficientNetB0 transfer learning
python -m src.training.trainer --config config.yaml --model efficientnet

# Graph Convolutional Network on skeleton
python -m src.training.trainer --config config.yaml --model skeleton_gcn

# CNN with attention
python -m src.training.trainer --config config.yaml --model attention_cnn

# Ensemble
python -m src.training.trainer --config config.yaml --model ensemble
```

### Hyperparameter Tuning

```bash
python -m src.training.hyperparameter_tuning --model custom_cnn --trials 20
```

---

## 3. Evaluation

### Evaluate a Saved Model

```python
from src.evaluation.metrics import ModelEvaluator
from tensorflow import keras

model = keras.models.load_model('results/models/custom_cnn_best.h5')
evaluator = ModelEvaluator(model, class_names=loader.class_names)
report = evaluator.full_report(X_test, y_test)
print(report)
```
### Ablation Study

```bash
python -m src.evaluation.ablation_study --config config.yaml
```

### Robustness Testing

```bash
python -m src.evaluation.robustness_testing --model results/models/custom_cnn_best.h5
```

### Visualize Results

```python
from src.evaluation.visualizations import ResultVisualizer

viz = ResultVisualizer()
viz.plot_training_history('results/logs/custom_cnn_history.json')
viz.plot_confusion_matrix(y_true, y_pred, class_names=loader.class_names)
viz.plot_model_comparison(results_dict)
```

---

## 4. Web Interface

### Convert Model to TensorFlow.js

```bash
python web/model_converter.py \
    --model_path results/models/custom_cnn_best.h5 \
    --output_dir web/tfjs_model
```

### Run Locally

Open `web/index.html` in a browser (Chrome recommended), allow camera access, and start detecting signs.

### Deploy to GitHub Pages

```bash
git add web/
git commit -m "Add web interface"
git push origin main
```
Enable GitHub Pages in repository Settings → Pages → Source: main branch / `web/` folder.

---

## 5. Jupyter Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Dataset analysis, class distribution, sample visualization |
| `02_model_training.ipynb` | Interactive training with real-time plots |
| `03_evaluation_analysis.ipynb` | Model comparison, confusion matrices, metrics |
| `04_web_deployment.ipynb` | Model conversion and deployment walkthrough |

```bash
jupyter notebook notebooks/
```

---

## 6. Configuration

Edit `config.yaml` to change hyperparameters:

```yaml
training:
  batch_size: 32      # Increase for faster training (needs more GPU RAM)
  epochs: 50          # Maximum epochs (early stopping may halt sooner)
  learning_rate: 0.001
  patience: 10        # Early stopping patience

model:
  dropout_rate: 0.5
  dense_units: 512
```

---

## 7. Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_models.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html
```
