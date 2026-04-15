"""Logging configuration for ASL detection project."""

import os
import logging
import sys
from typing import Optional


def setup_logging(log_level: str = "INFO",
                  log_file: Optional[str] = None,
                  log_dir: str = "results/logs") -> logging.Logger:
    """Configure logging for the project.

    Sets up console handler with colored output (if loguru available)
    and optional file handler.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional log filename. If provided, saves to log_dir.
        log_dir: Directory for log files.

    Returns:
        Root logger instance.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    root_logger.info(f"Logging initialized at level {log_level}")
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Named logger instance.
    """
    return logging.getLogger(name)
