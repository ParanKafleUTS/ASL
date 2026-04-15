"""Configuration management for ASL detection project."""

import os
import logging
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def get_model_config(config: Dict, model_name: str) -> Dict:
    """Get model-specific configuration.

    Args:
        config: Full configuration dictionary.
        model_name: Name of the model (e.g., 'custom_cnn', 'mobilenet').

    Returns:
        Model-specific configuration dictionary.
    """
    models_cfg = config.get("models", {})
    if model_name not in models_cfg:
        logger.warning(f"No config found for model '{model_name}', using defaults.")
        return {}
    return models_cfg[model_name]


def get_path(config: Dict, key: str) -> str:
    """Get a path from the config, ensuring the directory exists.

    Args:
        config: Configuration dictionary.
        key: Path key (e.g., 'data_raw', 'models').

    Returns:
        Resolved path string.
    """
    paths = config.get("paths", {})
    path = paths.get(key, key)
    os.makedirs(path, exist_ok=True)
    return path


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge two configuration dictionaries.

    Values in override take precedence over base.

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        Merged configuration dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
