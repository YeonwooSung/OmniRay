"""Faster Whisper STT model wrapper."""

import logging
from typing import Any, Dict

import numpy as np

from omniray.config.schemas import FasterWhisperConfig
from omniray.stt.base import BaseSTTModel

logger = logging.getLogger(__name__)


class FasterWhisperModel(BaseSTTModel):
    """Faster Whisper STT model implementation.

    Faster Whisper is a reimplementation of OpenAI's Whisper model using CTranslate2,
    which is up to 4x faster than the original implementation with similar accuracy.
    """

    def __init__(self, config: FasterWhisperConfig, **kwargs):
        """Initialize Faster Whisper model.

        Args:
            config: Faster Whisper configuration
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.config = config
        self.model = None

    def load_model(self) -> None:
        """Load Faster Whisper model."""
        try:
            from faster_whisper import WhisperModel

            logger.info(
                f"Loading Faster Whisper model: {self.config.model_size} "
                f"(device={self.config.device}, compute_type={self.config.compute_type})"
            )

            # Determine device
            device = self.config.device
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model
            self.model = WhisperModel(
                self.config.model_size,
                device=device,
                compute_type=self.config.compute_type,
            )

            logger.info(
                f"Faster Whisper model loaded successfully on {device} "
                f"with {self.config.compute_type} precision"
            )

        except ImportError:
            raise ImportError(
                "faster-whisper package is required. "
                "Install it with: pip install faster-whisper"
            )

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, **kwargs
    ) -> Dict[str, Any]:
        """Transcribe audio using Faster Whisper.

        Args:
            audio: Audio data as numpy array (1D float32)
            sample_rate: Sample rate of audio
            **kwargs: Additional transcription arguments (override config)

        Returns:
            Dictionary containing:
                - text: Full transcription text
                - segments: List of segments with timestamps and metadata
                - language: Detected language (if not specified)
        """
        if self.model is None:
            self.load_model()

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Prepare transcription parameters
        transcribe_kwargs = {
            "language": kwargs.get("language", self.config.language),
            "task": kwargs.get("task", self.config.task),
            "beam_size": kwargs.get("beam_size", self.config.beam_size),
            "vad_filter": kwargs.get("vad_filter", self.config.vad_filter),
        }

        # Add VAD parameters if specified
        if self.config.vad_parameters is not None:
            transcribe_kwargs["vad_parameters"] = self.config.vad_parameters

        # Remove None values
        transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if v is not None}

        # Run transcription
        segments_generator, info = self.model.transcribe(audio, **transcribe_kwargs)

        # Collect segments
        segments = []
        full_text_parts = []

        for segment in segments_generator:
            segment_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
            }

            # Add words if available
            if hasattr(segment, "words") and segment.words:
                segment_dict["words"] = [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]

            segments.append(segment_dict)
            full_text_parts.append(segment.text.strip())

        # Combine full text
        full_text = " ".join(full_text_parts)

        return {
            "text": full_text,
            "segments": segments,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "num_segments": len(segments),
        }
