"""Pseudo label generation for emotion analysis."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
from ray.data import Dataset

from omniray.config.schemas import InferenceConfig, ModelType, VideoConfig
from omniray.core.pipeline import VideoInferencePipeline
from omniray.labeling.label_storage import LabelStorage, LabelFormat

logger = logging.getLogger(__name__)


class EmotionPseudoLabeler:
    """Generate pseudo labels for emotion analysis on videos.
    
    This class processes videos to extract emotion labels which can be used as
    pseudo labels for training emotion recognition models.
    """

    def __init__(
        self,
        detector: str = "retinaface",
        au_model: str = "xgb",
        emotion_model: str = "resmasknet",
        confidence_threshold: float = 0.5,
        num_gpus: int = 0,
        num_cpus: int = 4,
    ):
        """Initialize emotion pseudo labeler.

        Args:
            detector: Face detector ('retinaface', 'mtcnn', 'faceboxes')
            au_model: Action Unit model ('svm', 'xgb', 'rf')
            emotion_model: Emotion model ('svm', 'resmasknet')
            confidence_threshold: Minimum confidence for valid labels
            num_gpus: Number of GPUs to use for inference
            num_cpus: Number of CPUs to use for inference
        """
        self.detector = detector
        self.au_model = au_model
        self.emotion_model = emotion_model
        self.confidence_threshold = confidence_threshold
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus


    def generate_labels(
        self,
        video_path: str,
        output_dir: str,
        label_format: LabelFormat = LabelFormat.JSON,
        batch_size: int = 16,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        save_frames: bool = False,
    ) -> Dict[str, Any]:
        """Generate pseudo labels for a video.

        Args:
            video_path: Path to input video
            output_dir: Directory to save labels and frames
            label_format: Format for saving labels (JSON, CSV, PARQUET)
            batch_size: Batch size for processing
            frame_skip: Process every Nth frame
            max_frames: Maximum frames to process
            save_frames: Whether to save frames with detected faces

        Returns:
            Dictionary with generation statistics and paths
        """
        logger.info(f"Generating pseudo labels for: {video_path}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Configure video processing
        video_config = VideoConfig(
            video_path=video_path,
            batch_size=batch_size,
            frame_skip=frame_skip,
            max_frames=max_frames,
        )

        # Configure inference
        inference_config = InferenceConfig(
            model_type=ModelType.EMOTION_ANALYSIS,
            video_config=video_config,
            ray_options={
                "num_cpus": self.num_cpus,
                "num_gpus": self.num_gpus,
            },
        )

        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        try:
            # Run inference pipeline
            pipeline = VideoInferencePipeline(inference_config)
            results_dataset = pipeline.run()

            # Process and filter results
            labels = self._process_results(results_dataset, save_frames, output_path)

            # Save labels
            storage = LabelStorage(output_dir=str(output_path))
            label_path = storage.save_labels(
                labels=labels,
                format=label_format,
                filename_prefix=Path(video_path).stem,
            )

            # Generate statistics
            stats = self._generate_statistics(labels)
            stats["label_path"] = str(label_path)
            stats["video_path"] = video_path
            stats["total_frames_processed"] = len(labels)

            logger.info(f"Label generation completed. Stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error generating labels: {e}")
            raise

    def _process_results(
        self,
        results_dataset: Dataset,
        save_frames: bool,
        output_path: Path,
    ) -> List[Dict[str, Any]]:
        """Process inference results into pseudo labels.

        Args:
            results_dataset: Ray dataset with inference results
            save_frames: Whether to save frame images
            output_path: Output directory path

        Returns:
            List of processed label dictionaries
        """
        labels = []
        results_list = results_dataset.take_all()

        for result in results_list:
            frame_idx = result["frame_idx"]
            predictions = result["predictions"]

            # Process each detected face
            for face in predictions.get("faces", []):
                # Filter by confidence if available
                if "confidence" in face and face["confidence"] is not None:
                    if face["confidence"] < self.confidence_threshold:
                        continue

                # Extract dominant emotion
                if "emotions" in face:
                    emotions = face["emotions"]
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    
                    label_entry = {
                        "frame_idx": frame_idx,
                        "face_id": face["face_id"],
                        "dominant_emotion": dominant_emotion[0],
                        "emotion_confidence": dominant_emotion[1],
                        "all_emotions": emotions,
                        "bbox": face.get("bbox"),
                        "face_confidence": face.get("confidence"),
                    }

                    # Add action units if available
                    if "action_units" in face:
                        label_entry["action_units"] = face["action_units"]

                    labels.append(label_entry)

        logger.info(f"Processed {len(labels)} pseudo labels from results")
        return labels

    def _generate_statistics(self, labels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about the generated labels.

        Args:
            labels: List of label dictionaries

        Returns:
            Dictionary with statistics
        """
        if not labels:
            return {
                "total_labels": 0,
                "unique_frames": 0,
                "emotion_distribution": {},
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
            }

        # Count emotion distribution
        emotion_counts = {}
        confidences = []

        for label in labels:
            emotion = label["dominant_emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            confidences.append(label["emotion_confidence"])

        stats = {
            "total_labels": len(labels),
            "unique_frames": len(set(l["frame_idx"] for l in labels)),
            "emotion_distribution": emotion_counts,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
        }

        return stats

    def generate_labels_batch(
        self,
        video_paths: List[str],
        output_dir: str,
        label_format: LabelFormat = LabelFormat.JSON,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate pseudo labels for multiple videos.

        Args:
            video_paths: List of video paths
            output_dir: Base output directory
            label_format: Format for saving labels
            **kwargs: Additional arguments for generate_labels

        Returns:
            List of statistics for each video
        """
        all_stats = []

        for video_path in video_paths:
            try:
                video_name = Path(video_path).stem
                video_output_dir = Path(output_dir) / video_name
                
                stats = self.generate_labels(
                    video_path=video_path,
                    output_dir=str(video_output_dir),
                    label_format=label_format,
                    **kwargs,
                )
                all_stats.append(stats)

            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                all_stats.append({
                    "video_path": video_path,
                    "error": str(e),
                })

        return all_stats
