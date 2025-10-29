"""Example: Run inference pipeline from YAML configuration file."""

import argparse
import logging

import ray
import yaml

from omniray import InferenceConfig, VideoInferencePipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config_from_yaml(yaml_path: str) -> InferenceConfig:
    """Load inference configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        InferenceConfig object
    """
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return InferenceConfig(**config_dict)


def main():
    """Run inference pipeline from YAML configuration."""
    parser = argparse.ArgumentParser(description="Run OmniRay inference from YAML config")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config_from_yaml(args.config)

    # Initialize Ray
    ray.init(ignore_reinit_error=True, **config.ray_options)

    try:
        # Create and run pipeline
        pipeline = VideoInferencePipeline(config)
        logger.info(f"Starting {config.model_type.value} pipeline...")

        results = pipeline.run()

        # Get summary
        summary = pipeline.get_summary()
        logger.info("=" * 60)
        logger.info("Pipeline Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        # Display sample results
        sample_results = results.take(3)
        logger.info("\nSample results from first 3 frames:")
        for result in sample_results:
            logger.info(f"Frame {result['frame_idx']} @ {result['timestamp']:.2f}s")
            logger.info(f"  Predictions: {result['predictions']}")

        logger.info(f"\nResults saved to: {config.output_path}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
