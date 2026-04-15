"""Script to download the ASL hand sign dataset from Kaggle using kagglehub."""

import os
import sys
import shutil
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID = "jeyasrisenthil/hand-signs-asl-hand-sign-data"


def download_dataset(output_dir: str = "data/raw") -> str:
    """Download the ASL hand sign dataset using kagglehub.

    kagglehub caches the download and returns the local path. This function
    also copies the data into *output_dir* so the rest of the codebase has a
    stable, predictable location.

    Args:
        output_dir: Directory where the dataset will be copied.

    Returns:
        Absolute path to the dataset directory inside output_dir.
    """
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub package not installed. Run: pip install kagglehub")
        sys.exit(1)

    logger.info(f"Downloading dataset: {DATASET_ID}")
    cached_path = kagglehub.dataset_download(DATASET_ID)
    logger.info(f"Dataset cached at: {cached_path}")

    abs_output = os.path.abspath(output_dir)
    abs_cached = os.path.abspath(cached_path)

    if abs_cached != abs_output:
        os.makedirs(abs_output, exist_ok=True)
        logger.info(f"Copying dataset to: {abs_output}")
        for item in os.listdir(abs_cached):
            src = os.path.join(abs_cached, item)
            dst = os.path.join(abs_output, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    logger.info(f"Dataset ready at: {abs_output}")
    return abs_output


def verify_dataset(data_dir: str = "data/raw") -> bool:
    """Verify that the dataset directory contains at least one class subdirectory.

    The function searches both the top-level directory and common sub-folder
    names used by this dataset (e.g. ``Train/``, ``train/``).

    Args:
        data_dir: Root directory of the downloaded dataset.

    Returns:
        True if class folders are found, False otherwise.
    """
    if not os.path.isdir(data_dir):
        logger.warning(f"Dataset directory not found: {data_dir}")
        return False

    # Directories to probe for class subfolders
    candidate_roots = [data_dir] + [
        os.path.join(data_dir, sub)
        for sub in ("Train", "train", "Test", "test",
                    "asl_alphabet_train", "asl_alphabet_test")
    ]

    for root in candidate_roots:
        if not os.path.isdir(root):
            continue
        class_dirs = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
        ]
        if class_dirs:
            logger.info(
                f"Dataset verification passed. "
                f"Found {len(class_dirs)} class folder(s) under '{root}'."
            )
            return True

    logger.warning(f"No class subdirectories found under '{data_dir}'.")
    return False


def main() -> None:
    """Main entry point for downloading the dataset."""
    parser = argparse.ArgumentParser(
        description=f"Download '{DATASET_ID}' from Kaggle via kagglehub"
    )
    parser.add_argument(
        "--output-dir", default="data/raw",
        help="Directory to copy the dataset into (default: data/raw)"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only check if the dataset already exists; do not download"
    )
    args = parser.parse_args()

    if args.verify_only:
        exists = verify_dataset(args.output_dir)
        sys.exit(0 if exists else 1)

    download_dataset(args.output_dir)
    verify_dataset(args.output_dir)


if __name__ == "__main__":
    main()
