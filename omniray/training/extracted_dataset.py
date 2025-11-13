"""Dataset for loading pre-extracted frames."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)


class ExtractedFrameDataset(TorchDataset):
    """PyTorch Dataset for pre-extracted emotion frames.
    
    This dataset loads frames that have been extracted and saved as images,
    which is more efficient than loading from video on-the-fly.
    """

    def __init__(
        self,
        frames_dir: str,
        metadata_dir: Optional[str] = None,
        transform: Optional[Any] = None,
        target_size: Tuple[int, int] = (224, 224),
        emotion_to_idx: Optional[Dict[str, int]] = None,
        min_confidence: Optional[float] = None,
        image_format: str = "jpg",
    ):
        """Initialize extracted frame dataset.

        Args:
            frames_dir: Directory containing extracted frames
            metadata_dir: Optional directory containing metadata JSON files
            transform: Optional transform to apply to frames
            target_size: Target size for frames (width, height)
            emotion_to_idx: Mapping from emotion names to class indices
            min_confidence: Minimum confidence threshold for filtering
            image_format: Image file format (jpg, png)
        """
        self.frames_dir = Path(frames_dir)
        self.metadata_dir = Path(metadata_dir) if metadata_dir else None
        self.transform = transform
        self.target_size = target_size
        self.image_format = image_format
        self.min_confidence = min_confidence

        # Scan and build file list
        self.samples = self._scan_frames()

        # Filter by confidence if needed
        if min_confidence is not None and self.metadata_dir is not None:
            self.samples = self._filter_by_confidence(self.samples, min_confidence)

        # Create emotion to index mapping
        if emotion_to_idx is None:
            unique_emotions = sorted(set(s["emotion"] for s in self.samples))
            self.emotion_to_idx = {
                emotion: idx for idx, emotion in enumerate(unique_emotions)
            }
        else:
            self.emotion_to_idx = emotion_to_idx

        self.idx_to_emotion = {
            idx: emotion for emotion, idx in self.emotion_to_idx.items()
        }

        logger.info(
            f"Initialized ExtractedFrameDataset with {len(self.samples)} samples, "
            f"{len(self.emotion_to_idx)} emotion classes"
        )

    def _scan_frames(self) -> List[Dict[str, Any]]:
        """Scan frames directory and build sample list.

        Returns:
            List of sample dictionaries
        """
        samples = []

        # Check if organized by emotion folders
        emotion_dirs = [
            d for d in self.frames_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ]

        if emotion_dirs:
            # Organized by emotion
            logger.info("Detected emotion-organized directory structure")
            for emotion_dir in emotion_dirs:
                emotion = emotion_dir.name
                for image_file in emotion_dir.glob(f"*.{self.image_format}"):
                    samples.append(
                        {
                            "image_path": str(image_file),
                            "emotion": emotion,
                            "metadata_path": self._get_metadata_path(image_file)
                            if self.metadata_dir
                            else None,
                        }
                    )
        else:
            # All in one directory - need metadata to get emotions
            logger.info("Flat directory structure, loading from metadata")
            if not self.metadata_dir:
                raise ValueError(
                    "Metadata directory required for flat directory structure"
                )

            for image_file in self.frames_dir.glob(f"*.{self.image_format}"):
                metadata_path = self._get_metadata_path(image_file)
                if metadata_path and metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    samples.append(
                        {
                            "image_path": str(image_file),
                            "emotion": metadata.get("emotion"),
                            "metadata_path": str(metadata_path),
                        }
                    )

        logger.info(f"Found {len(samples)} frames")
        return samples

    def _get_metadata_path(self, image_file: Path) -> Optional[Path]:
        """Get metadata path for an image file.

        Args:
            image_file: Image file path

        Returns:
            Metadata file path or None
        """
        if not self.metadata_dir:
            return None

        metadata_filename = image_file.name.replace(
            f".{self.image_format}", ".json"
        )
        return self.metadata_dir / metadata_filename

    def _filter_by_confidence(
        self, samples: List[Dict[str, Any]], min_confidence: float
    ) -> List[Dict[str, Any]]:
        """Filter samples by confidence threshold.

        Args:
            samples: List of samples
            min_confidence: Minimum confidence

        Returns:
            Filtered samples
        """
        filtered_samples = []

        for sample in samples:
            metadata_path = sample.get("metadata_path")
            if metadata_path and Path(metadata_path).exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                confidence = metadata.get("emotion_confidence", 0)
                if confidence >= min_confidence:
                    filtered_samples.append(sample)
            else:
                # No metadata, keep sample
                filtered_samples.append(sample)

        logger.info(
            f"Filtered {len(samples)} -> {len(filtered_samples)} samples "
            f"(min_confidence={min_confidence})"
        )
        return filtered_samples

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Get item from dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (image, label_idx, metadata)
        """
        sample = self.samples[idx]

        # Load image
        image_path = sample["image_path"]
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, self.target_size)

        # Apply transform if available
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default normalization
            image = image.astype(np.float32) / 255.0

        # Get emotion label
        emotion = sample["emotion"]
        label_idx = self.emotion_to_idx.get(emotion, 0)

        # Load metadata if available
        metadata = {"image_path": image_path, "emotion": emotion}
        
        if sample.get("metadata_path"):
            metadata_path = sample["metadata_path"]
            if Path(metadata_path).exists():
                with open(metadata_path, "r") as f:
                    full_metadata = json.load(f)
                metadata.update(full_metadata)

        return image, label_idx, metadata

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in dataset.

        Returns:
            Dictionary mapping emotions to counts
        """
        distribution = {}
        for sample in self.samples:
            emotion = sample["emotion"]
            distribution[emotion] = distribution.get(emotion, 0) + 1

        return distribution

    def get_class_weights(self) -> np.ndarray:
        """Calculate class weights for imbalanced datasets.

        Returns:
            Array of class weights
        """
        distribution = self.get_class_distribution()
        total_samples = len(self.samples)
        num_classes = len(self.emotion_to_idx)

        weights = np.zeros(num_classes)
        for emotion, count in distribution.items():
            idx = self.emotion_to_idx[emotion]
            weights[idx] = total_samples / (num_classes * count)

        return weights
