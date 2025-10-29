"""Video frame loading with Ray Data."""

import logging
from typing import Iterator, Optional

import cv2
import numpy as np
import ray
from ray.data import Dataset

from omniray.config.schemas import VideoConfig

logger = logging.getLogger(__name__)


class VideoFrameDataset:
    """Video frame dataset using Ray Data."""

    def __init__(self, config: VideoConfig):
        """Initialize video frame dataset.

        Args:
            config: Video configuration
        """
        self.config = config
        self.video_path = config.video_path

    def _read_frames(self) -> Iterator[dict]:
        """Read frames from video file.

        Yields:
            Dictionary containing frame data and metadata
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Loading video: {self.video_path} "
            f"(fps={fps:.2f}, total_frames={total_frames})"
        )

        frame_idx = 0
        frames_processed = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Skip frames if configured
                if frame_idx % self.config.frame_skip != 0:
                    frame_idx += 1
                    continue

                # Resize if target size is specified
                if self.config.target_size is not None:
                    frame = cv2.resize(frame, self.config.target_size)

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield {
                    "frame": frame,
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps if fps > 0 else 0.0,
                    "video_path": self.video_path,
                }

                frames_processed += 1
                frame_idx += 1

                # Check if we've reached max frames
                if (
                    self.config.max_frames is not None
                    and frames_processed >= self.config.max_frames
                ):
                    break

        finally:
            cap.release()

        logger.info(f"Processed {frames_processed} frames from video")

    def create_dataset(self) -> Dataset:
        """Create Ray Dataset from video frames.

        Returns:
            Ray Dataset containing video frames
        """
        # Create Ray Dataset from the frame generator
        ds = ray.data.from_items([{"video_path": self.video_path}])

        # Flat map to expand each video into frames
        ds = ds.flat_map(lambda _: self._read_frames())

        # Repartition for better parallelism if needed
        if self.config.batch_size:
            num_blocks = max(1, ds.count() // self.config.batch_size)
            ds = ds.repartition(num_blocks)

        return ds


def load_video_frames(config: VideoConfig) -> Dataset:
    """Load video frames as a Ray Dataset.

    Args:
        config: Video configuration

    Returns:
        Ray Dataset containing video frames

    Example:
        >>> from omniray.config import VideoConfig
        >>> from omniray.data import load_video_frames
        >>> config = VideoConfig(video_path="video.mp4", batch_size=32)
        >>> ds = load_video_frames(config)
        >>> print(ds.take(1))
    """
    dataset = VideoFrameDataset(config)
    return dataset.create_dataset()
