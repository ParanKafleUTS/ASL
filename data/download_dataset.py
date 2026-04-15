"""Script to download the Sign Language MNIST dataset from Kaggle."""

import os
import sys
import zipfile
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_kaggle_dataset(dataset: str = "datamunge/sign-language-mnist",
                            output_dir: str = "data/raw") -> None:
    """Download dataset from Kaggle using the Kaggle API.

    Args:
        dataset: Kaggle dataset identifier (owner/dataset-name).
        output_dir: Directory where the dataset will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import kaggle
    except ImportError:
        logger.error("Kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    # Check for Kaggle credentials
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json):
        logger.error(
            "Kaggle credentials not found. Please:\n"
            "1. Create a Kaggle account\n"
            "2. Go to Account -> Create API Token\n"
            "3. Save kaggle.json to ~/.kaggle/kaggle.json\n"
            "4. Run: chmod 600 ~/.kaggle/kaggle.json"
        )
        sys.exit(1)

    logger.info(f"Downloading dataset: {dataset}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=output_dir, unzip=True)
    logger.info(f"Dataset downloaded to: {output_dir}")

    # List downloaded files
    files = os.listdir(output_dir)
    logger.info(f"Downloaded files: {files}")


def verify_dataset(data_dir: str = "data/raw") -> bool:
    """Verify that the Sign Language MNIST dataset files exist.

    Args:
        data_dir: Directory containing the dataset files.

    Returns:
        True if dataset files are present, False otherwise.
    """
    required_files = ["sign_mnist_train.csv", "sign_mnist_test.csv"]
    for fname in required_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            logger.warning(f"Missing file: {fpath}")
            return False
    logger.info("Dataset verification passed.")
    return True


def main() -> None:
    """Main entry point for downloading the dataset."""
    parser = argparse.ArgumentParser(description="Download Sign Language MNIST dataset")
    parser.add_argument("--dataset", default="datamunge/sign-language-mnist",
                        help="Kaggle dataset identifier")
    parser.add_argument("--output-dir", default="data/raw",
                        help="Output directory for the dataset")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify if dataset exists, don't download")
    args = parser.parse_args()

    if args.verify_only:
        exists = verify_dataset(args.output_dir)
        sys.exit(0 if exists else 1)

    download_kaggle_dataset(args.dataset, args.output_dir)
    verify_dataset(args.output_dir)


if __name__ == "__main__":
    main()
