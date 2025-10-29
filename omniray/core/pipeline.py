"""Main inference pipeline orchestrator using Ray."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import ray
from ray.data import Dataset

from omniray.config.schemas import InferenceConfig, ModelType
from omniray.data.video_loader import load_video_frames
from omniray.models.custom import CustomModel
from omniray.models.detection import ObjectDetectionModel
from omniray.models.emotion import EmotionAnalysisModel


logger = logging.getLogger(__name__)


class VideoInferencePipeline:
    """Ray-based video inference pipeline."""

    def __init__(self, config: InferenceConfig):
        """Initialize inference pipeline.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.model = None
        self.results = None

    def _create_model(self):
        """Create model instance based on configuration."""
        model_type = self.config.model_type

        if model_type == ModelType.OBJECT_DETECTION:
            logger.info("Creating object detection model")
            self.model = ObjectDetectionModel()

        elif model_type == ModelType.EMOTION_ANALYSIS:
            logger.info("Creating emotion analysis model")
            self.model = EmotionAnalysisModel()

        elif model_type == ModelType.CUSTOM:
            logger.info("Creating custom model")
            if self.config.custom_model_config is None:
                raise ValueError("custom_model_config is required for CUSTOM model type")
            self.model = CustomModel(self.config.custom_model_config)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load the model
        self.model.load_model()

    def _inference_fn(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Ray map function for batch inference.

        Args:
            batch: Batch dictionary containing frame data

        Returns:
            Batch with predictions added
        """
        # This will be called within Ray workers
        # The model needs to be loaded within each worker
        if not hasattr(self, "_worker_model"):
            self._create_model()
            self._worker_model = self.model

        return self._worker_model(batch)

    def run(self) -> Dataset:
        """Run the inference pipeline.

        Returns:
            Ray Dataset containing frames with predictions

        Example:
            >>> from omniray.config import InferenceConfig, VideoConfig, ModelType
            >>> from omniray.core import VideoInferencePipeline
            >>>
            >>> config = InferenceConfig(
            ...     model_type=ModelType.OBJECT_DETECTION,
            ...     video_config=VideoConfig(video_path="video.mp4")
            ... )
            >>> pipeline = VideoInferencePipeline(config)
            >>> results = pipeline.run()
            >>> print(results.take(1))
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(**self.config.ray_options)
            logger.info("Ray initialized")

        # Load video frames as Ray Dataset
        logger.info(f"Loading video: {self.config.video_config.video_path}")
        dataset = load_video_frames(self.config.video_config)

        # Create model (this will be serialized to Ray workers)
        self._create_model()

        # Run inference using Ray Data
        logger.info("Running inference on video frames")

        # Use map_batches for efficient batch processing
        # (ray data distributed inference)
        results_dataset = dataset.map_batches(
            lambda batch: self._inference_fn(batch),
            batch_size=self.config.video_config.batch_size,
            num_gpus=self.config.ray_options.get("num_gpus", 0),
        )

        self.results = results_dataset

        # Save results if output path is specified
        if self.config.output_path:
            self.save_results(self.config.output_path)

        logger.info("Inference completed")
        return results_dataset

    def save_results(self, output_path: str) -> None:
        """Save inference results to file.

        Args:
            output_path: Path to save results (supports .json, .parquet)
        """
        if self.results is None:
            raise ValueError("No results to save. Run inference first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to: {output_path}")

        if output_path.suffix == ".json":
            # Convert to list and save as JSON
            results_list = self.results.take_all()
            with open(output_path, "w") as f:
                json.dump(results_list, f, indent=2, default=str)

        elif output_path.suffix == ".parquet":
            # Save as Parquet
            self.results.write_parquet(str(output_path))

        else:
            raise ValueError(
                f"Unsupported output format: {output_path.suffix}. "
                f"Supported formats: .json, .parquet"
            )

        logger.info(f"Results saved to: {output_path}")

    def get_results(self) -> List[Dict[str, Any]]:
        """Get inference results as a list.

        Returns:
            List of dictionaries containing frame data and predictions
        """
        if self.results is None:
            raise ValueError("No results available. Run inference first.")

        return self.results.take_all()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of inference results.

        Returns:
            Dictionary containing summary statistics
        """
        if self.results is None:
            raise ValueError("No results available. Run inference first.")

        results_list = self.results.take_all()

        summary = {
            "total_frames": len(results_list),
            "video_path": self.config.video_config.video_path,
            "model_type": self.config.model_type.value,
        }

        # Add model-specific summaries
        if self.config.model_type == ModelType.OBJECT_DETECTION:
            total_detections = sum(r["predictions"]["num_detections"] for r in results_list)
            summary["total_detections"] = total_detections
            summary["avg_detections_per_frame"] = (
                total_detections / len(results_list) if results_list else 0
            )

        elif self.config.model_type == ModelType.EMOTION_ANALYSIS:
            total_faces = sum(r["predictions"]["num_faces"] for r in results_list)
            summary["total_faces"] = total_faces
            summary["avg_faces_per_frame"] = total_faces / len(results_list) if results_list else 0

        return summary
