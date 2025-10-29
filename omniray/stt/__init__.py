"""STT (Speech-to-Text) module for OmniRay."""

from omniray.stt.base import BaseSTTModel
from omniray.stt.faster_whisper import FasterWhisperModel
from omniray.stt.pipeline import STTInferencePipeline

__all__ = [
    "BaseSTTModel",
    "FasterWhisperModel",
    "STTInferencePipeline",
]
