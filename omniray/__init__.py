"""
OmniRay: Ray-based scalable AI-intensive system.
"""

__VERSION__ = "0.2.1"

# Video inference
from omniray.config import InferenceConfig, ModelType, VideoConfig
from omniray.core import VideoInferencePipeline

# STT (Speech-to-Text)
from omniray.config import AudioConfig, FasterWhisperConfig, STTInferenceConfig, STTModelType
from omniray.stt import STTInferencePipeline


__all__ = [
    # Video inference
    "InferenceConfig",
    "ModelType",
    "VideoConfig",
    "VideoInferencePipeline",
    # STT
    "AudioConfig",
    "FasterWhisperConfig",
    "STTInferenceConfig",
    "STTModelType",
    "STTInferencePipeline",
]
