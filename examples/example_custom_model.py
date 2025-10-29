"""Example: Custom model inference on video using YAML configuration."""

import logging

import ray

from omniray import InferenceConfig, ModelType, VideoConfig, VideoInferencePipeline
from omniray.config import CustomModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run custom model inference on a video file."""
    # Configure video input
    video_config = VideoConfig(
        video_path="path/to/your/video.mp4",  # Change this to your video path
        batch_size=32,
        frame_skip=1,
        max_frames=None,
    )

    # Configure custom model
    # This example assumes you have a custom model class
    custom_model_config = CustomModelConfig(
        model_class="mymodels.MyCustomModel",  # Replace with your model class
        model_path="path/to/model/weights.pth",  # Optional: path to model weights
        model_kwargs={
            "num_classes": 10,
            "input_size": 224,
            # Add any other model initialization parameters
        },
        preprocessing={
            "resize": [224, 224],
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        postprocessing={
            "threshold": 0.5,
        },
    )

    # Configure inference pipeline
    inference_config = InferenceConfig(
        model_type=ModelType.CUSTOM,
        video_config=video_config,
        custom_model_config=custom_model_config,
        ray_options={
            "num_cpus": 4,
            "num_gpus": 0,  # Set to 1 if you have GPU
        },
        output_path="results/custom_results.json",
    )

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Create and run pipeline
        pipeline = VideoInferencePipeline(inference_config)
        logger.info("Starting custom model inference pipeline...")

        results = pipeline.run()

        # Get summary
        summary = pipeline.get_summary()
        logger.info(f"Pipeline completed. Summary: {summary}")

        # Display sample results
        sample_results = results.take(5)
        logger.info("Sample results from first 5 frames:")
        for i, result in enumerate(sample_results):
            logger.info(f"Frame {result['frame_idx']}: {result['predictions']}")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
