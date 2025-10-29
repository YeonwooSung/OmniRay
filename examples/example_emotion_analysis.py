"""Example: Emotion analysis on video using py-feat."""

import logging
import sys
import ray
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # To import from parent directory

from omniray import InferenceConfig, ModelType, VideoConfig, VideoInferencePipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(video_path: str):
    """Run emotion analysis on a video file."""
    # Configure video input
    video_config = VideoConfig(
        video_path=video_path,  # Change this to your video path
        batch_size=16,  # Smaller batch size for emotion analysis
        frame_skip=5,  # Process every 5th frame (emotion doesn't change rapidly)
        max_frames=100,  # Limit frames for faster testing
    )

    # Configure inference pipeline
    inference_config = InferenceConfig(
        model_type=ModelType.EMOTION_ANALYSIS,
        video_config=video_config,
        ray_options={
            "num_cpus": 4,
            "num_gpus": 0,  # Set to 1 if you have GPU
        },
        output_path="results/emotion_results.json",
    )

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Create and run pipeline
        pipeline = VideoInferencePipeline(inference_config)
        logger.info("Starting emotion analysis pipeline...")

        results = pipeline.run()

        # Get summary
        summary = pipeline.get_summary()
        logger.info(f"Pipeline completed. Summary: {summary}")

        # Display sample results
        sample_results = results.take(5)
        logger.info("Sample results from first 5 frames:")
        for i, result in enumerate(sample_results):
            logger.info(
                f"Frame {result['frame_idx']}: " f"{result['predictions']['num_faces']} faces detected"
            )
            if result["predictions"]["num_faces"] > 0:
                for face in result["predictions"]["faces"]:
                    if "emotions" in face:
                        # Find dominant emotion
                        emotions = face["emotions"]
                        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                        logger.info(f"  Face {face['face_id']}: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    video_path = input("Enter the path to the video file for emotion analysis: ").strip()
    main(video_path)
