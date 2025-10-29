"""Configuration schemas for OmniRay pipelines."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Supported model types."""

    OBJECT_DETECTION = "object_detection"
    EMOTION_ANALYSIS = "emotion_analysis"
    CUSTOM = "custom"


class VideoConfig(BaseModel):
    """Video loading configuration."""

    video_path: str = Field(..., description="Path to the video file")
    batch_size: int = Field(default=32, description="Batch size for frame processing")
    frame_skip: int = Field(default=1, description="Process every Nth frame (1 = all frames)")
    max_frames: Optional[int] = Field(default=None, description="Maximum number of frames to process")
    target_size: Optional[tuple[int, int]] = Field(
        default=None, description="Target size (width, height) for frames"
    )


class CustomModelConfig(BaseModel):
    """Configuration for custom models."""

    model_class: str = Field(..., description="Fully qualified class name (e.g., 'mymodule.MyModel')")
    model_path: Optional[str] = Field(default=None, description="Path to model weights")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Model initialization kwargs")
    preprocessing: Optional[Dict[str, Any]] = Field(
        default=None, description="Preprocessing configuration"
    )
    postprocessing: Optional[Dict[str, Any]] = Field(
        default=None, description="Postprocessing configuration"
    )


class InferenceConfig(BaseModel):
    """Main inference pipeline configuration."""

    model_type: ModelType = Field(..., description="Type of model to use")
    video_config: VideoConfig = Field(..., description="Video loading configuration")
    custom_model_config: Optional[CustomModelConfig] = Field(
        default=None, description="Custom model configuration (required if model_type is CUSTOM)"
    )
    ray_options: Dict[str, Any] = Field(
        default_factory=dict, description="Ray-specific options (e.g., num_gpus, num_cpus)"
    )
    output_path: Optional[str] = Field(default=None, description="Path to save inference results")

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        if self.model_type == ModelType.CUSTOM and self.custom_model_config is None:
            raise ValueError("custom_model_config is required when model_type is CUSTOM")
