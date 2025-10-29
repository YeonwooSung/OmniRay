"""
OmniRay: Ray-based scalable AI-intensive system.
"""

__VERSION__ = "0.2.0"

from omniray.config import InferenceConfig, ModelType, VideoConfig
from omniray.core import VideoInferencePipeline

__all__ = [
    "InferenceConfig",
    "ModelType",
    "VideoConfig",
    "VideoInferencePipeline",
]
