"""Example: Object detection on video using YOLOv8."""

import logging

import ray

from omniray import InferenceConfig, ModelType, VideoConfig, VideoInferencePipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run object detection on a video file."""
    # Configure video input
    video_config = VideoConfig(
        video_path="path/to/your/video.mp4",  # Change this to your video path
        batch_size=32,
        frame_skip=1,  # Process every frame
        max_frames=None,  # Process all frames (set to a number for testing)
    )

    # Configure inference pipeline
    inference_config = InferenceConfig(
        model_type=ModelType.OBJECT_DETECTION,
        video_config=video_config,
        ray_options={
            "num_cpus": 4,
            "num_gpus": 0,  # Set to 1 if you have GPU
        },
        output_path="results/detection_results.json",
    )

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Create and run pipeline
        pipeline = VideoInferencePipeline(inference_config)
        logger.info("Starting object detection pipeline...")

        results = pipeline.run()

        # Get summary
        summary = pipeline.get_summary()
        logger.info(f"Pipeline completed. Summary: {summary}")

        # Display sample results
        sample_results = results.take(5)
        logger.info("Sample results from first 5 frames:")
        for i, result in enumerate(sample_results):
            logger.info(
                f"Frame {result['frame_idx']}: "
                f"{result['predictions']['num_detections']} objects detected"
            )
            if result["predictions"]["num_detections"] > 0:
                logger.info(f"  Classes: {result['predictions']['class_names']}")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
