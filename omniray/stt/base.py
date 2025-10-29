"""Base STT model interface for OmniRay."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class BaseSTTModel(ABC):
    """Abstract base class for all STT models."""

    def __init__(self, **kwargs):
        """Initialize STT model with optional kwargs."""
        self.model_kwargs = kwargs

    @abstractmethod
    def load_model(self) -> None:
        """Load the STT model into memory."""
        pass

    @abstractmethod
    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, **kwargs
    ) -> Dict[str, Any]:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (1D float32)
            sample_rate: Sample rate of audio
            **kwargs: Additional transcription arguments

        Returns:
            Dictionary containing transcription results with at minimum:
                - text: Full transcription text
                - segments: List of segments with timestamps
        """
        pass

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of audio chunks using Ray.

        Args:
            batch: Batch dictionary containing 'audio' and 'sample_rate' keys

        Returns:
            Batch dictionary with added 'transcription' key
        """
        audio = batch["audio"]
        sample_rate = batch.get("sample_rate", 16000)

        transcription = self.transcribe(audio, sample_rate)

        return {
            **batch,
            "transcription": transcription,
        }
