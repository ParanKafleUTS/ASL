# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip 21+
- (Optional) CUDA-enabled GPU for faster training
- Git

## 1. Clone the Repository

```bash
git clone https://github.com/ParanKafleUTS/ASL.git
cd ASL
```

## 2. Create and Activate a Virtual Environment

```bash
# Using venv
python -m venv asl_env
source asl_env/bin/activate       # Linux/macOS
asl_env\Scripts\activate          # Windows

# Or using conda
conda create -n asl python=3.10
conda activate asl
```

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU Support (Optional but Recommended)

For NVIDIA GPUs with CUDA 11.x:

```bash
pip install tensorflow-gpu==2.12.0
```

For CUDA 12.x or later:

```bash
pip install tensorflow[and-cuda]
```

Verify GPU is detected:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## 4. Install the Package (Editable Mode)

```bash
pip install -e .
```

## 5. Download the Dataset

This project uses the **ASL Alphabet** dataset by grassknoted on Kaggle.
Download it with `kagglehub` (no manual credentials file required):

```bash
python data/download_dataset.py
```

`kagglehub` will prompt for your Kaggle credentials on first use and cache them automatically.

Alternatively you can download directly in Python:

```python
import kagglehub
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Path to dataset files:", path)
```

Expected structure after download (copied into `data/raw/`):

```
data/raw/
└── asl_alphabet_train/
    └── asl_alphabet_train/
        ├── A/        (≈3 000 images)
        ├── B/
        ├── ...
        ├── Z/
        ├── del/
        ├── nothing/
        └── space/
```

> **Note:** The dataset contains **29 classes** — the 26 ASL letters (A–Z,
> including J and Z as static poses) plus `del`, `nothing`, and `space`.
> Each class has approximately 3 000 colour images at 200 × 200 pixels.

## 6. Verify Installation

```bash
python -c "import src; print('Package installed successfully')"
python -m pytest tests/ -v --tb=short
```

## Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: tensorflow` | Run `pip install tensorflow` |
| CUDA out of memory | Reduce `batch_size` in `config.yaml` |
| `kagglehub: command not found` | Run `pip install kagglehub` |
| OpenCV camera issues | Install `pip install opencv-python-headless` |

## Environment Variables

```bash
export ASL_DATA_DIR=/path/to/data       # Custom data directory
export ASL_RESULTS_DIR=/path/to/results # Custom results directory
export TF_CPP_MIN_LOG_LEVEL=2           # Suppress TensorFlow logs
```
