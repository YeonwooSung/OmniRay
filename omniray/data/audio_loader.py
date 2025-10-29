"""Audio loading with Ray Data for STT tasks."""

import logging
import subprocess
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import ray
from ray.data import Dataset

from omniray.config.schemas import AudioConfig

logger = logging.getLogger(__name__)


def extract_audio_with_ffmpeg(
    input_path: str, sample_rate: int = 16000, mono: bool = True
) -> np.ndarray:
    """Extract audio from video/audio file using ffmpeg.

    Args:
        input_path: Path to input file
        sample_rate: Target sample rate
        mono: Convert to mono if True

    Returns:
        Audio array as float32 numpy array
    """
    import tempfile

    # Create temporary wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-ac",
            "1" if mono else "2",  # Audio channels
            "-ar",
            str(sample_rate),  # Sample rate
            "-f",
            "wav",  # Output format
            "-y",  # Overwrite
            tmp_path,
        ]

        # Run ffmpeg
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

        # Load the audio file
        from pydub import AudioSegment

        audio = AudioSegment.from_wav(tmp_path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Normalize to [-1, 1]
        if audio.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio.sample_width == 4:  # 32-bit
            samples = samples / 2147483648.0

        return samples

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to extract audio with ffmpeg: {e.stderr.decode()}"
        )
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


class AudioChunkDataset:
    """Audio chunk dataset using Ray Data."""

    def __init__(self, config: AudioConfig):
        """Initialize audio chunk dataset.

        Args:
            config: Audio configuration
        """
        self.config = config
        self.audio_path = config.audio_path

    def _load_and_chunk_audio(self) -> Iterator[dict]:
        """Load audio file and split into chunks.

        Yields:
            Dictionary containing audio chunk data and metadata
        """
        logger.info(f"Loading audio from: {self.audio_path}")

        # Extract audio using ffmpeg
        audio_array = extract_audio_with_ffmpeg(
            self.audio_path, sample_rate=self.config.sample_rate, mono=True
        )

        total_duration = len(audio_array) / self.config.sample_rate
        logger.info(
            f"Loaded audio: duration={total_duration:.2f}s, "
            f"sample_rate={self.config.sample_rate}Hz"
        )

        # Determine how much audio to process
        if self.config.max_duration_s is not None:
            max_samples = int(self.config.max_duration_s * self.config.sample_rate)
            audio_array = audio_array[:max_samples]
            total_duration = min(total_duration, self.config.max_duration_s)

        # Calculate chunk parameters
        chunk_samples = int(self.config.chunk_length_s * self.config.sample_rate)
        total_samples = len(audio_array)

        chunk_idx = 0
        start_sample = 0

        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = audio_array[start_sample:end_sample]

            start_time = start_sample / self.config.sample_rate
            end_time = end_sample / self.config.sample_rate
            duration = end_time - start_time

            yield {
                "audio": chunk,
                "chunk_idx": chunk_idx,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "sample_rate": self.config.sample_rate,
                "audio_path": self.audio_path,
            }

            start_sample = end_sample
            chunk_idx += 1

        logger.info(f"Created {chunk_idx} audio chunks")

    def create_dataset(self) -> Dataset:
        """Create Ray Dataset from audio chunks.

        Returns:
            Ray Dataset containing audio chunks
        """
        # Create Ray Dataset from audio chunks
        ds = ray.data.from_items([{"audio_path": self.audio_path}])

        # Flat map to expand audio file into chunks
        ds = ds.flat_map(lambda _: self._load_and_chunk_audio())

        # Repartition for better parallelism if needed
        if self.config.batch_size:
            num_chunks = ds.count()
            num_blocks = max(1, num_chunks // self.config.batch_size)
            ds = ds.repartition(num_blocks)

        return ds


def load_audio_chunks(config: AudioConfig) -> Dataset:
    """Load audio file as chunked Ray Dataset.

    Args:
        config: Audio configuration

    Returns:
        Ray Dataset containing audio chunks

    Example:
        >>> from omniray.config import AudioConfig
        >>> from omniray.data import load_audio_chunks
        >>> config = AudioConfig(audio_path="audio.mp3", chunk_length_s=30.0)
        >>> ds = load_audio_chunks(config)
        >>> print(ds.take(1))
    """
    dataset = AudioChunkDataset(config)
    return dataset.create_dataset()
