"""Convert trained Keras models to TensorFlow.js format for web deployment.

Run this script after training models to convert them for use in the web
interface. Requires the tensorflowjs package.

Usage:
    python web/model_converter.py --model-name custom_cnn
    python web/model_converter.py --all
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "custom_cnn":    "results/models/custom_cnn_best.h5",
    "mobilenet":     "results/models/mobilenet_v2_asl_best.h5",
    "efficientnet":  "results/models/efficientnet_b0_asl_best.h5",
    "attention_cnn": "results/models/attention_cnn_cbam_best.h5",
    "ensemble":      "results/models/ensemble_best.h5",
}

WEB_MODELS_DIR = "web/models"


def convert_model(model_name: str, keras_path: str, output_dir: str) -> bool:
    """Convert a Keras model to TensorFlow.js format.

    Args:
        model_name: Model display name.
        keras_path: Path to the .h5 Keras model file.
        output_dir: Output directory for TFJS model files.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    if not os.path.exists(keras_path):
        logger.warning(f"Model file not found: {keras_path}. Skipping {model_name}.")
        return False

    try:
        import tensorflowjs as tfjs
    except ImportError:
        logger.error(
            "tensorflowjs not installed. Run: pip install tensorflowjs"
        )
        return False

    out_path = os.path.join(output_dir, model_name)
    os.makedirs(out_path, exist_ok=True)

    logger.info(f"Converting {model_name} from {keras_path}...")
    try:
        tfjs.converters.save_keras_model(
            __import__("tensorflow").keras.models.load_model(keras_path),
            out_path
        )
        logger.info(f"  Saved to: {out_path}")
        return True
    except Exception as e:
        logger.error(f"Conversion failed for {model_name}: {e}")
        return False


def main() -> None:
    """Main entry point for model conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Keras models to TensorFlow.js format"
    )
    parser.add_argument("--model-name", choices=list(MODEL_CONFIGS.keys()),
                        help="Name of the model to convert")
    parser.add_argument("--all", action="store_true",
                        help="Convert all available models")
    parser.add_argument("--output-dir", default=WEB_MODELS_DIR,
                        help="Output directory for TFJS models")
    args = parser.parse_args()

    if not args.model_name and not args.all:
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        models_to_convert = MODEL_CONFIGS.items()
    else:
        models_to_convert = [(args.model_name, MODEL_CONFIGS[args.model_name])]

    success_count = 0
    for name, keras_path in models_to_convert:
        if convert_model(name, keras_path, args.output_dir):
            success_count += 1

    logger.info(f"Converted {success_count}/{len(list(models_to_convert))} models.")
    logger.info(f"TFJS models saved in: {args.output_dir}")
    logger.info("Update the model paths in web/app.js if needed.")


if __name__ == "__main__":
    main()
