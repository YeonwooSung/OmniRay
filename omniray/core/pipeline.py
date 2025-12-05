"""Main inference pipeline orchestrator using Ray."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import ray
from ray.data import Dataset

from omniray.config.schemas import InferenceConfig, ModelType
from omniray.data.video_loader import load_video_frames


logger = logging.getLogger(__name__)


class EmotionModelActor:
    """Actor class for emotion model inference - loads model once per worker."""

    def __init__(self, detector: str = "retinaface", au_model: str = "xgb", emotion_model: str = "resmasknet"):
        self.detector = detector
        self.au_model = au_model
        self.emotion_model = emotion_model
        self.model = None

    def _load_model(self):
        """Load model on first call."""
        if self.model is None:
            from omniray.models.emotion import EmotionAnalysisModel
            self.model = EmotionAnalysisModel(
                detector=self.detector,
                au_model=self.au_model,
                emotion_model=self.emotion_model,
            )
            self.model.load_model()

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of frames."""
        self._load_model()
        return self.model(batch)


class ObjectDetectionModelActor:
    """Actor class for object detection model inference."""
    
    def __init__(self):
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            from omniray.models.detection import ObjectDetectionModel
            self.model = ObjectDetectionModel()
            self.model.load_model()
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self._load_model()
        return self.model(batch)


class CustomModelActor:
    """Actor class for custom model inference."""
    
    def __init__(self, custom_model_config: dict):
        self.custom_model_config = custom_model_config
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            from omniray.models.custom import CustomModel
            self.model = CustomModel(self.custom_model_config)
            self.model.load_model()
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self._load_model()
        return self.model(batch)


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


    def _get_model_actor_class(self):
        """Get the appropriate model actor class based on configuration.
        
        Returns:
            Tuple of (ActorClass, kwargs) for the model
        """
        model_type = self.config.model_type

        if model_type == ModelType.OBJECT_DETECTION:
            logger.info("Creating object detection model")
            return ObjectDetectionModelActor, {}

        elif model_type == ModelType.EMOTION_ANALYSIS:
            logger.info("Creating emotion analysis model")
            return EmotionModelActor, {}

        elif model_type == ModelType.CUSTOM:
            logger.info("Creating custom model")
            if self.config.custom_model_config is None:
                raise ValueError("custom_model_config is required for CUSTOM model type")
            return CustomModelActor, {"custom_model_config": self.config.custom_model_config}

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

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

        # Get the model actor class (lightweight - doesn't load actual model yet)
        actor_class, actor_kwargs = self._get_model_actor_class()

        # Run inference using Ray Data with class-based actor
        # The model will be loaded once per worker, not serialized from driver
        logger.info("Running inference on video frames")

        # ray data를 통해서 batch 단위로 actor를 호출하여 inference 수행
        # Use batch_format="numpy" to ensure frames are passed as numpy arrays
        results_dataset = dataset.map_batches(
            actor_class,
            batch_size=self.config.video_config.batch_size,
            batch_format="numpy",  # Ensure numpy format for frame data
            num_gpus=self.config.ray_options.get("num_gpus", 0),
            concurrency=self.config.ray_options.get("num_cpus", 4),
            fn_constructor_kwargs=actor_kwargs if actor_kwargs else None,
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
