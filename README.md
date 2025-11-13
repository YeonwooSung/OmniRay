# OmniRay

Ray-based scalable AI inference and training system for video and beyond.

OmniRay leverages Ray Data and Ray's distributed computing capabilities to provide efficient, scalable AI inference pipelines for video processing, with plans to expand to language models, speech-to-text, and other ML tasks.

## Features

- **Video Frame Processing**: Load and process video files efficiently using Ray Data
  - Object Detection (YOLOv8)
  - Emotion Analysis (py-feat)
  - Custom Models (YAML configuration)
- **Speech-to-Text (STT)**: Transcribe audio/video with Faster Whisper
  - Multi-language support (90+ languages)
  - Automatic language detection
  - Translation to English
  - Export to SRT/VTT subtitles
- **Pseudo Labeling & Training**: Generate pseudo labels and train models with Ray
  - Automated emotion label generation from videos
  - Training dataset preparation utilities
  - Distributed training with Ray Train
  - Label storage in multiple formats (JSON, CSV, Parquet)
- **Scalable Ray-based Inference**: Distributed processing across CPUs/GPUs
- **Flexible Configuration**: Python API or YAML configuration files
- **Extensible Architecture**: Easy to add new model types and data sources

## Installation

```bash
# Install from source
cd OmniRay
pip install -e .

# Or using uv
uv pip install -e .
```

## Quick Start

### 1. Object Detection

```python
from omniray import InferenceConfig, ModelType, VideoConfig, VideoInferencePipeline

# Configure the pipeline
config = InferenceConfig(
    model_type=ModelType.OBJECT_DETECTION,
    video_config=VideoConfig(
        video_path="video.mp4",
        batch_size=32,
        frame_skip=1,
    ),
    output_path="results/detections.json"
)

# Run inference
pipeline = VideoInferencePipeline(config)
results = pipeline.run()

# Get summary
summary = pipeline.get_summary()
print(summary)
```

### 2. Emotion Analysis

```python
from omniray import InferenceConfig, ModelType, VideoConfig, VideoInferencePipeline

config = InferenceConfig(
    model_type=ModelType.EMOTION_ANALYSIS,
    video_config=VideoConfig(
        video_path="video.mp4",
        batch_size=16,
        frame_skip=5,  # Process every 5th frame
    ),
    output_path="results/emotions.json"
)

pipeline = VideoInferencePipeline(config)
results = pipeline.run()
```

### 3. Custom Model

```python
from omniray import InferenceConfig, ModelType, VideoConfig
from omniray.config import CustomModelConfig

custom_config = CustomModelConfig(
    model_class="mymodels.MyModel",
    model_path="weights.pth",
    model_kwargs={"num_classes": 10},
    preprocessing={
        "resize": [224, 224],
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
)

config = InferenceConfig(
    model_type=ModelType.CUSTOM,
    video_config=VideoConfig(video_path="video.mp4"),
    custom_model_config=custom_config,
)

pipeline = VideoInferencePipeline(config)
results = pipeline.run()
```

### 4. Speech-to-Text (STT)

```python
from omniray import (
    AudioConfig,
    FasterWhisperConfig,
    STTInferenceConfig,
    STTModelType,
    STTInferencePipeline,
)

# Configure STT pipeline
config = STTInferenceConfig(
    model_type=STTModelType.FASTER_WHISPER,
    audio_config=AudioConfig(
        audio_path="audio.mp3",
        chunk_length_s=30.0,
        batch_size=8,
    ),
    faster_whisper_config=FasterWhisperConfig(
        model_size="base",  # tiny, base, small, medium, large-v3
        language="en",  # or None for auto-detect
        task="transcribe",  # or "translate" for English translation
    ),
    output_path="results/transcription.json"
)

# Run transcription
pipeline = STTInferencePipeline(config)
results = pipeline.run()

# Get full transcription
text = pipeline.get_full_transcription()
print(text)

# Save as subtitles
pipeline.save_results("results/subtitles.srt")
```

### 5. Using YAML Configuration

```bash
# Create a config file (see examples/config_object_detection.yaml)
python examples/run_from_config.py examples/config_object_detection.yaml
```

## Project Structure

```
omniray/
├── config/          # Configuration schemas
├── core/            # Video pipeline orchestration
├── data/            # Data loaders (video, audio)
│   ├── video_loader.py  # Video frame loading
│   └── audio_loader.py  # Audio chunk loading
├── models/          # Vision model wrappers
│   ├── base.py      # Base model interface
│   ├── detection.py # Object detection
│   ├── emotion.py   # Emotion analysis
│   └── custom.py    # Custom model loader
├── stt/             # Speech-to-Text models
│   ├── base.py          # Base STT interface
│   ├── faster_whisper.py # Faster Whisper
│   └── pipeline.py      # STT pipeline
├── labeling/        # Pseudo labeling utilities
│   ├── emotion_labeler.py  # Emotion pseudo labeler
│   └── label_storage.py    # Label storage & management
└── training/        # Training pipelines
    ├── __init__.py      # Dataset preparation
    └── trainer.py       # Ray-based distributed training
```

## Examples

See the [examples/](examples/) directory for complete examples:

**Video Processing:**
- [example_object_detection.py](examples/example_object_detection.py) - Object detection with YOLOv8
- [example_emotion_analysis.py](examples/example_emotion_analysis.py) - Facial emotion analysis
- [example_custom_model.py](examples/example_custom_model.py) - Custom model integration

**Speech-to-Text:**
- [example_stt_faster_whisper.py](examples/example_stt_faster_whisper.py) - Basic STT with Faster Whisper
- [example_stt_multilanguage.py](examples/example_stt_multilanguage.py) - Multi-language transcription

**Pseudo Labeling & Training:**
- [example_pseudo_labeling_training.py](examples/example_pseudo_labeling_training.py) - Full pseudo labeling and training pipeline
- [example_pseudo_labeling_from_config.py](examples/example_pseudo_labeling_from_config.py) - Run from YAML config

**Configuration:**
- [run_from_config.py](examples/run_from_config.py) - Run from YAML config files

## Documentation

- [STT Guide](docs/stt_guide.md) - Comprehensive Speech-to-Text usage guide
- [Pseudo Labeling Guide](docs/pseudo_labeling_guide.md) - Pseudo labeling and training guide
- [Full Frame Extraction Guide](docs/full_frame_extraction_guide.md) - Complete frame extraction pipeline
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions
- [Helpful Resources](docs/helpful_resources.md) - Additional resources and references

## Roadmap

- [x] Video inference pipeline (Object detection, Emotion analysis)
- [x] Speech-to-text (STT) pipeline with Faster Whisper
- [x] Pseudo labeling for emotion recognition
- [x] Training pipeline integration with Ray Train
- [ ] Language model inference support (vLLM integration)
- [ ] Multi-modal pipelines (Video + Audio fusion)
- [ ] ETL data preprocessing pipelines
- [ ] Streaming inference support
- [ ] Additional STT backends (OpenAI Whisper, Wav2Vec2)
- [ ] Active learning with pseudo labels

## References

- [Scaling Pinterest ML Infrastructure with Ray: From Training to End-to-End ML Pipelines](https://medium.com/pinterest-engineering/scaling-pinterest-ml-infrastructure-with-ray-from-training-to-end-to-end-ml-pipelines-4038b9e837a0)
- [Ray Batch Inference at Pinterest (Part 3)](https://medium.com/pinterest-engineering/ray-batch-inference-at-pinterest-part-3-4faeb652e385)
- [Batch Predictions in Ray](https://docs.ray.io/en/latest/ray-core/examples/batch_prediction.html)
