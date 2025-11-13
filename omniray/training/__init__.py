"""Dataset preparation utilities for training."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset as TorchDataset

from omniray.labeling.label_storage import LabelStorage

logger = logging.getLogger(__name__)

# Export extracted dataset if available
try:
    from omniray.training.extracted_dataset import ExtractedFrameDataset
    __all__ = ["EmotionDataset", "DatasetBuilder", "ExtractedFrameDataset"]
except ImportError:
    __all__ = ["EmotionDataset", "DatasetBuilder"]


class EmotionDataset(TorchDataset):
    """PyTorch Dataset for emotion recognition training.
    
    This dataset loads frames and their corresponding emotion labels
    generated as pseudo labels from videos.
    """

    def __init__(
        self,
        labels: List[Dict[str, Any]],
        video_path: str,
        transform: Optional[Any] = None,
        target_size: Tuple[int, int] = (224, 224),
        emotion_to_idx: Optional[Dict[str, int]] = None,
    ):
        """Initialize emotion dataset.

        Args:
            labels: List of label dictionaries from pseudo labeling
            video_path: Path to the source video
            transform: Optional transform to apply to frames
            target_size: Target size for frames (width, height)
            emotion_to_idx: Mapping from emotion names to class indices
        """
        self.labels = labels
        self.video_path = video_path
        self.transform = transform
        self.target_size = target_size

        # Create emotion to index mapping if not provided
        if emotion_to_idx is None:
            unique_emotions = sorted(set(label["dominant_emotion"] for label in labels))
            self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
        else:
            self.emotion_to_idx = emotion_to_idx

        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}

        # Cache video capture
        self.video_capture = None
        self._frame_cache = {}

        logger.info(
            f"Initialized EmotionDataset with {len(labels)} samples, "
            f"{len(self.emotion_to_idx)} emotion classes"
        )

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Get item from dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (image, label_idx, metadata)
        """
        label = self.labels[idx]
        
        # Load frame
        frame = self._load_frame(label["frame_idx"])
        
        # Crop face if bbox is available
        if label.get("bbox") is not None:
            frame = self._crop_face(frame, label["bbox"])
        
        # Resize
        frame = cv2.resize(frame, self.target_size)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transform if available
        if self.transform is not None:
            frame = self.transform(frame)
        else:
            # Default normalization
            frame = frame.astype(np.float32) / 255.0
        
        # Get emotion label
        emotion = label["dominant_emotion"]
        label_idx = self.emotion_to_idx[emotion]
        
        # Metadata
        metadata = {
            "frame_idx": label["frame_idx"],
            "face_id": label["face_id"],
            "emotion_confidence": label["emotion_confidence"],
            "video_path": self.video_path,
        }
        
        return frame, label_idx, metadata

    def _load_frame(self, frame_idx: int) -> np.ndarray:
        """Load frame from video.

        Args:
            frame_idx: Frame index

        Returns:
            Frame as numpy array
        """
        # Check cache
        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx].copy()
        
        # Open video if not already open
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(self.video_path)
        
        # Set frame position
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = self.video_capture.read()
        
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {self.video_path}")
        
        # Cache frame (limit cache size)
        if len(self._frame_cache) < 100:
            self._frame_cache[frame_idx] = frame.copy()
        
        return frame

    def _crop_face(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop face from frame using bounding box.

        Args:
            frame: Full frame
            bbox: Bounding box [x, y, width, height]

        Returns:
            Cropped face region
        """
        x, y, w, h = [int(v) for v in bbox]
        
        # Add padding
        padding = int(max(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        return frame[y:y+h, x:x+w]

    def close(self):
        """Close video capture and clear cache."""
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self._frame_cache.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class DatasetBuilder:
    """Build training datasets from pseudo labels."""

    def __init__(self, label_storage: Optional[LabelStorage] = None):
        """Initialize dataset builder.

        Args:
            label_storage: Optional LabelStorage instance
        """
        self.label_storage = label_storage

    def build_dataset(
        self,
        label_path: str,
        video_path: str,
        transform: Optional[Any] = None,
        min_confidence: Optional[float] = None,
        emotions: Optional[List[str]] = None,
        **kwargs,
    ) -> EmotionDataset:
        """Build emotion dataset from labels.

        Args:
            label_path: Path to label file
            video_path: Path to video file
            transform: Optional transform
            min_confidence: Minimum confidence threshold for filtering
            emotions: List of emotions to include
            **kwargs: Additional arguments for EmotionDataset

        Returns:
            EmotionDataset instance
        """
        # Load labels
        if self.label_storage is None:
            storage = LabelStorage(output_dir=str(Path(label_path).parent))
        else:
            storage = self.label_storage
        
        labels = storage.load_labels(label_path)
        
        # Filter labels
        if min_confidence is not None or emotions is not None:
            labels = storage.filter_labels(
                labels,
                min_confidence=min_confidence,
                emotions=emotions,
            )
        
        logger.info(f"Building dataset with {len(labels)} samples")
        
        # Create dataset
        dataset = EmotionDataset(
            labels=labels,
            video_path=video_path,
            transform=transform,
            **kwargs,
        )
        
        return dataset

    def build_datasets_from_multiple_videos(
        self,
        label_video_pairs: List[Tuple[str, str]],
        transform: Optional[Any] = None,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        **kwargs,
    ) -> Tuple[TorchDataset, TorchDataset, TorchDataset]:
        """Build train/val/test datasets from multiple videos.

        Args:
            label_video_pairs: List of (label_path, video_path) tuples
            transform: Optional transform
            split_ratio: (train, val, test) split ratio
            **kwargs: Additional arguments for EmotionDataset

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        from torch.utils.data import ConcatDataset, random_split

        # Build individual datasets
        datasets = []
        for label_path, video_path in label_video_pairs:
            dataset = self.build_dataset(
                label_path=label_path,
                video_path=video_path,
                transform=transform,
                **kwargs,
            )
            datasets.append(dataset)
        
        # Concatenate all datasets
        full_dataset = ConcatDataset(datasets)
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(total_size * split_ratio[0])
        val_size = int(total_size * split_ratio[1])
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
        )
        
        logger.info(
            f"Split dataset into train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )
        
        return train_dataset, val_dataset, test_dataset

    def export_for_training(
        self,
        label_path: str,
        video_path: str,
        output_dir: str,
        extract_frames: bool = True,
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Export labels and frames for training.

        Args:
            label_path: Path to label file
            video_path: Path to video file
            output_dir: Output directory for exported data
            extract_frames: Whether to extract and save frames
            min_confidence: Minimum confidence threshold

        Returns:
            Export statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and filter labels
        storage = LabelStorage(output_dir=str(Path(label_path).parent))
        labels = storage.load_labels(label_path)
        
        if min_confidence is not None:
            labels = storage.filter_labels(labels, min_confidence=min_confidence)
        
        # Extract frames if requested
        if extract_frames:
            frames_dir = output_path / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            video_capture = cv2.VideoCapture(video_path)
            extracted_count = 0
            
            for label in labels:
                frame_idx = label["frame_idx"]
                face_id = label["face_id"]
                emotion = label["dominant_emotion"]
                
                # Create emotion directory
                emotion_dir = frames_dir / emotion
                emotion_dir.mkdir(exist_ok=True)
                
                # Read frame
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video_capture.read()
                
                if ret:
                    # Crop face if bbox available
                    if label.get("bbox") is not None:
                        x, y, w, h = [int(v) for v in label["bbox"]]
                        frame = frame[y:y+h, x:x+w]
                    
                    # Save frame
                    frame_filename = f"frame_{frame_idx}_face_{face_id}.jpg"
                    frame_path = emotion_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    extracted_count += 1
            
            video_capture.release()
            logger.info(f"Extracted {extracted_count} frames to {frames_dir}")
        
        # Save filtered labels
        filtered_label_path = output_path / "filtered_labels.json"
        storage_out = LabelStorage(output_dir=str(output_path))
        storage_out.save_labels(labels, filename_prefix="filtered")
        
        stats = {
            "total_labels": len(labels),
            "output_dir": str(output_path),
            "frames_extracted": extracted_count if extract_frames else 0,
        }
        
        return stats
