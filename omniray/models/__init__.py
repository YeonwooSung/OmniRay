"""Model wrappers for OmniRay."""

from omniray.models.base import BaseModel
from omniray.models.custom import CustomModel
from omniray.models.detection import ObjectDetectionModel
from omniray.models.emotion import EmotionAnalysisModel

__all__ = [
    "BaseModel",
    "CustomModel",
    "ObjectDetectionModel",
    "EmotionAnalysisModel",
]
