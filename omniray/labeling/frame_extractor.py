"""Frame extraction utilities for pseudo labeling."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract and save frames with labels for training."""

    def __init__(
        self,
        output_dir: str,
        save_format: str = "jpg",
        jpeg_quality: int = 95,
        organize_by_emotion: bool = True,
        include_metadata: bool = True,
    ):
        """Initialize frame extractor.

        Args:
            output_dir: Output directory for extracted frames
            save_format: Image format (jpg, png)
            jpeg_quality: JPEG quality (1-100)
            organize_by_emotion: Organize frames in emotion folders
            include_metadata: Save metadata JSON files
        """
        self.output_dir = Path(output_dir)
        self.save_format = save_format.lower()
        self.jpeg_quality = jpeg_quality
        self.organize_by_emotion = organize_by_emotion
        self.include_metadata = include_metadata

        # Create output directory
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        if include_metadata:
            self.metadata_dir = self.output_dir / "metadata"
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def extract_and_save_frames(
        self,
        video_path: str,
        labels: List[Dict[str, Any]],
        save_full_frames: bool = False,
    ) -> Dict[str, Any]:
        """Extract frames from video and save with labels.

        Args:
            video_path: Path to video file
            labels: List of label dictionaries
            save_full_frames: Save full frames instead of cropped faces

        Returns:
            Extraction statistics
        """
        video_path = Path(video_path)
        video_name = video_path.stem

        logger.info(f"Extracting frames from {video_path.name}")

        # Open video
        video_capture = cv2.VideoCapture(str(video_path))
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Statistics
        stats = {
            "total_frames_saved": 0,
            "frames_by_emotion": {},
            "video_name": video_name,
            "save_format": self.save_format,
        }

        # Group labels by frame_idx for efficient processing
        labels_by_frame = {}
        for label in labels:
            frame_idx = label["frame_idx"]
            if frame_idx not in labels_by_frame:
                labels_by_frame[frame_idx] = []
            labels_by_frame[frame_idx].append(label)

        # Process frames
        frame_indices = sorted(labels_by_frame.keys())
        
        for frame_idx in frame_indices:
            # Set video position
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_capture.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                continue

            # Process each face in this frame
            for label in labels_by_frame[frame_idx]:
                emotion = label["dominant_emotion"]
                face_id = label["face_id"]

                # Create emotion directory if needed
                if self.organize_by_emotion:
                    emotion_dir = self.frames_dir / emotion
                    emotion_dir.mkdir(exist_ok=True)
                    save_dir = emotion_dir
                else:
                    save_dir = self.frames_dir

                # Get image to save
                if save_full_frames:
                    image_to_save = frame.copy()
                else:
                    # Crop face region
                    if label.get("bbox") is not None:
                        image_to_save = self._crop_face_with_padding(
                            frame, label["bbox"]
                        )
                    else:
                        # No bbox, save full frame
                        image_to_save = frame.copy()

                # Generate filename
                filename = f"{video_name}_frame{frame_idx:06d}_face{face_id}.{self.save_format}"
                image_path = save_dir / filename

                # Save image
                if self.save_format == "jpg":
                    cv2.imwrite(
                        str(image_path),
                        image_to_save,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                else:  # png
                    cv2.imwrite(str(image_path), image_to_save)

                # Save metadata if requested
                if self.include_metadata:
                    metadata = {
                        "image_path": str(image_path.relative_to(self.output_dir)),
                        "frame_idx": frame_idx,
                        "face_id": face_id,
                        "emotion": emotion,
                        "emotion_confidence": label.get("emotion_confidence"),
                        "all_emotions": label.get("all_emotions"),
                        "bbox": label.get("bbox"),
                        "face_confidence": label.get("face_confidence"),
                        "action_units": label.get("action_units"),
                    }

                    metadata_filename = filename.replace(
                        f".{self.save_format}", ".json"
                    )
                    metadata_path = self.metadata_dir / metadata_filename

                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                # Update statistics
                stats["total_frames_saved"] += 1
                if emotion not in stats["frames_by_emotion"]:
                    stats["frames_by_emotion"][emotion] = 0
                stats["frames_by_emotion"][emotion] += 1

        video_capture.release()

        logger.info(
            f"Extracted {stats['total_frames_saved']} frames from {video_name}"
        )
        logger.info(f"Distribution: {stats['frames_by_emotion']}")

        return stats

    def _crop_face_with_padding(
        self, frame: np.ndarray, bbox: List[float], padding_ratio: float = 0.2
    ) -> np.ndarray:
        """Crop face from frame with padding.

        Args:
            frame: Full frame
            bbox: Bounding box [x, y, width, height]
            padding_ratio: Padding ratio relative to bbox size

        Returns:
            Cropped face region
        """
        x, y, w, h = [int(v) for v in bbox]

        # Add padding
        padding = int(max(w, h) * padding_ratio)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)

        return frame[y : y + h, x : x + w].copy()

    def create_dataset_index(
        self, save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an index of all extracted frames.

        Args:
            save_path: Optional path to save index JSON

        Returns:
            Dataset index dictionary
        """
        index = {
            "frames": [],
            "emotion_counts": {},
            "total_frames": 0,
        }

        # Scan frames directory
        if self.organize_by_emotion:
            # Organized by emotion folders
            for emotion_dir in self.frames_dir.iterdir():
                if not emotion_dir.is_dir():
                    continue

                emotion = emotion_dir.name
                emotion_count = 0

                for image_file in emotion_dir.glob(f"*.{self.save_format}"):
                    # Load metadata if available
                    metadata = None
                    if self.include_metadata:
                        metadata_file = (
                            self.metadata_dir
                            / image_file.name.replace(
                                f".{self.save_format}", ".json"
                            )
                        )
                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)

                    index["frames"].append(
                        {
                            "image_path": str(
                                image_file.relative_to(self.output_dir)
                            ),
                            "emotion": emotion,
                            "metadata": metadata,
                        }
                    )
                    emotion_count += 1

                index["emotion_counts"][emotion] = emotion_count
        else:
            # All frames in one directory
            for image_file in self.frames_dir.glob(f"*.{self.save_format}"):
                # Try to load metadata to get emotion
                emotion = None
                metadata = None

                if self.include_metadata:
                    metadata_file = (
                        self.metadata_dir
                        / image_file.name.replace(f".{self.save_format}", ".json")
                    )
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                            emotion = metadata.get("emotion")

                index["frames"].append(
                    {
                        "image_path": str(
                            image_file.relative_to(self.output_dir)
                        ),
                        "emotion": emotion,
                        "metadata": metadata,
                    }
                )

                if emotion:
                    index["emotion_counts"][emotion] = (
                        index["emotion_counts"].get(emotion, 0) + 1
                    )

        index["total_frames"] = len(index["frames"])

        # Save index if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(index, f, indent=2)
            logger.info(f"Dataset index saved to {save_path}")

        return index

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about extracted frames.

        Returns:
            Statistics dictionary
        """
        index = self.create_dataset_index()

        return {
            "total_frames": index["total_frames"],
            "emotion_distribution": index["emotion_counts"],
            "frames_directory": str(self.frames_dir),
            "metadata_directory": str(self.metadata_dir)
            if self.include_metadata
            else None,
        }
