# 🤟 ASL Hand Sign Detection

> **Postgraduate Research Project** — Master's-level American Sign Language recognition using skeleton-guided deep learning approaches.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Sign%20Language%20MNIST-purple)](https://www.kaggle.com/datamunge/sign-language-mnist)

---

## 📋 Project Overview

This repository contains a **production-ready, postgraduate-level** ASL hand sign detection system that:

- Trains and compares **4+ deep learning architectures**: Custom CNN, MobileNetV2, EfficientNetB0, and a novel **Graph Convolutional Network (GCN)** on hand skeletons
- Uses the **Sign Language MNIST dataset** (28×28 grayscale images, 24 ASL letter classes)
- Provides a **real-time web interface** for live hand sign detection via webcam
- Implements **rigorous evaluation**: bootstrap confidence intervals, McNemar's significance tests, ablation studies, and robustness testing
- Avoids MediaPipe dependency issues by using a **custom OpenCV-based skeleton extractor**

### Key Research Contributions

1. **Skeleton-GCN**: Novel application of Graph Convolutional Networks directly on hand keypoint graphs for ASL classification
2. **Attention-CNN**: Integration of CBAM (Channel + Spatial attention) into the CNN architecture
3. **Comprehensive Comparison**: Systematic evaluation of CNN vs. GCN vs. Transfer Learning approaches
4. **Robustness Analysis**: Testing under rotation, occlusion, noise, and low-light conditions

---

## 🗂️ Project Structure

```
ASL/
├── README.md               # This file
├── THESIS.md               # Thesis structure and research methodology
├── requirements.txt        # All Python dependencies
├── config.yaml             # Hyperparameters and settings
├── setup.py                # Package installation
│
├── data/
│   ├── raw/                # Downloaded dataset (CSV files)
│   ├── processed/          # Preprocessed data cache
│   └── download_dataset.py # Kaggle dataset downloader
│
├── src/
│   ├── data/               # Data loading, augmentation, utilities
│   ├── models/             # CNN, GCN, Transfer Learning, Attention, Ensemble
│   ├── training/           # Trainer, callbacks, hyperparameter tuning
│   ├── evaluation/         # Metrics, ablation, robustness, visualizations
│   └── utils/              # Config, logging utilities
│
├── notebooks/              # Jupyter exploration and training notebooks
├── web/                    # Real-time web interface (TensorFlow.js)
├── tests/                  # Unit tests for all modules
├── results/                # Saved models, logs, plots, metrics
└── docs/                   # API docs, installation guide, usage guide
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/ParanKafleUTS/ASL.git
cd ASL
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python data/download_dataset.py
```

### 3. Train Models

```bash
python -m src.training.trainer --model custom_cnn --epochs 50
# Or: jupyter notebook notebooks/02_model_training.ipynb
```

### 4. Launch Web Interface

```bash
python web/model_converter.py --all
python -m http.server 8000
# Navigate to: http://localhost:8000/web/
```

---

## 🧠 Model Architectures

| Model | Expected Accuracy | Parameters | Notes |
|-------|:-----------------:|:----------:|-------|
| Custom CNN | ~92-95% | ~500K | Fast, good baseline |
| MobileNetV2 | ~95-97% | ~3.5M | Mobile-friendly |
| EfficientNetB0 | ~95-98% | ~5.3M | High accuracy |
| Attention CNN (CBAM) | ~93-96% | ~600K | Attention-enhanced |
| Skeleton GCN | ~85-92% | ~200K | Novel graph approach |
| Ensemble | ~96-99% | — | Best overall |

---

## 🔬 Evaluation Framework

```python
from src.evaluation.metrics import ASLEvaluator
evaluator = ASLEvaluator(num_classes=24)
metrics = evaluator.compute_metrics(y_test, y_pred, y_proba)
est, lower, upper = evaluator.bootstrap_confidence_interval(y_test, y_pred)
print(f"Accuracy: {est:.4f} [{lower:.4f}, {upper:.4f}] (95% CI)")
result = evaluator.mcnemar_test(y_test, pred_model1, pred_model2)
```

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v --cov=src --cov-report=html
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [THESIS.md](THESIS.md) | Full thesis structure and research methodology |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Detailed setup instructions |
| [docs/USAGE.md](docs/USAGE.md) | Training, evaluation, and deployment guide |
| [docs/API.md](docs/API.md) | Code API reference |
| [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) | Research findings and notes |

---

## 📝 Citation

```bibtex
@mastersthesis{kafle2024asl,
  title     = {Skeleton-Guided ASL Recognition using Graph Neural Networks and Attention Mechanisms},
  author    = {Paran Kafle},
  school    = {University of Technology Sydney},
  year      = {2024},
  type      = {Master's Thesis}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.