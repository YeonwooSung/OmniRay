"""Configuration schemas for OmniRay pipelines."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Supported model types for vision tasks."""

    OBJECT_DETECTION = "object_detection"
    EMOTION_ANALYSIS = "emotion_analysis"
    CUSTOM = "custom"


class STTModelType(str, Enum):
    """Supported STT model types."""

    FASTER_WHISPER = "faster_whisper"
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


class AudioConfig(BaseModel):
    """Audio loading configuration."""

    audio_path: str = Field(..., description="Path to the audio/video file")
    batch_size: int = Field(default=16, description="Batch size for audio chunk processing")
    chunk_length_s: float = Field(default=30.0, description="Length of each audio chunk in seconds")
    sample_rate: int = Field(default=16000, description="Target sample rate for audio")
    max_duration_s: Optional[float] = Field(
        default=None, description="Maximum audio duration to process in seconds"
    )


class FasterWhisperConfig(BaseModel):
    """Configuration for Faster Whisper model."""

    model_size: str = Field(
        default="base",
        description="Model size: tiny, base, small, medium, large-v2, large-v3",
    )
    device: str = Field(default="auto", description="Device: auto, cpu, cuda")
    compute_type: str = Field(
        default="default",
        description="Compute type: default, int8, int8_float16, int8_float32, float16, float32",
    )
    language: Optional[str] = Field(default=None, description="Language code (e.g., 'en', 'ko')")
    task: str = Field(default="transcribe", description="Task: transcribe or translate")
    beam_size: int = Field(default=5, description="Beam size for decoding")
    vad_filter: bool = Field(default=True, description="Enable voice activity detection filter")
    vad_parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="VAD filter parameters"
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


class STTInferenceConfig(BaseModel):
    """STT inference pipeline configuration."""

    model_type: STTModelType = Field(..., description="Type of STT model to use")
    audio_config: AudioConfig = Field(..., description="Audio loading configuration")
    faster_whisper_config: Optional[FasterWhisperConfig] = Field(
        default=None,
        description="Faster Whisper configuration (required if model_type is FASTER_WHISPER)",
    )
    custom_model_config: Optional[CustomModelConfig] = Field(
        default=None, description="Custom model configuration (required if model_type is CUSTOM)"
    )
    ray_options: Dict[str, Any] = Field(
        default_factory=dict, description="Ray-specific options (e.g., num_gpus, num_cpus)"
    )
    output_path: Optional[str] = Field(default=None, description="Path to save transcription results")

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        if (
            self.model_type == STTModelType.FASTER_WHISPER
            and self.faster_whisper_config is None
        ):
            # Provide default config
            self.faster_whisper_config = FasterWhisperConfig()

        if self.model_type == STTModelType.CUSTOM and self.custom_model_config is None:
            raise ValueError("custom_model_config is required when model_type is CUSTOM")
