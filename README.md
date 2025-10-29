# OmniRay

Ray-based scalable AI inference and training system for video and beyond.

OmniRay leverages Ray Data and Ray's distributed computing capabilities to provide efficient, scalable AI inference pipelines for video processing, with plans to expand to language models, speech-to-text, and other ML tasks.

## Features

- **Video Frame Processing**: Load and process video files efficiently using Ray Data
- **Multiple Model Support**:
  - Object Detection (YOLOv8)
  - Emotion Analysis (py-feat)
  - Custom Models (YAML configuration)
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

### 4. Using YAML Configuration

```bash
# Create a config file (see examples/config_object_detection.yaml)
python examples/run_from_config.py examples/config_object_detection.yaml
```

## Project Structure

```
omniray/
├── config/          # Configuration schemas
├── core/            # Pipeline orchestration
├── data/            # Data loading (video, etc.)
└── models/          # Model wrappers
    ├── base.py      # Base model interface
    ├── detection.py # Object detection
    ├── emotion.py   # Emotion analysis
    └── custom.py    # Custom model loader
```

## Examples

See the [examples/](examples/) directory for complete examples:
- [example_object_detection.py](examples/example_object_detection.py)
- [example_emotion_analysis.py](examples/example_emotion_analysis.py)
- [example_custom_model.py](examples/example_custom_model.py)
- [run_from_config.py](examples/run_from_config.py)

## Roadmap

- [ ] Language model inference support
- [ ] Speech-to-text (STT) pipeline
- [ ] Training pipeline integration
- [ ] ETL data preprocessing pipelines
- [ ] Multi-modal fusion pipelines
- [ ] Streaming inference support

## References

- [Scaling Pinterest ML Infrastructure with Ray: From Training to End-to-End ML Pipelines](https://medium.com/pinterest-engineering/scaling-pinterest-ml-infrastructure-with-ray-from-training-to-end-to-end-ml-pipelines-4038b9e837a0)
- [Ray Batch Inference at Pinterest (Part 3)](https://medium.com/pinterest-engineering/ray-batch-inference-at-pinterest-part-3-4faeb652e385)
- [Batch Predictions in Ray](https://docs.ray.io/en/latest/ray-core/examples/batch_prediction.html)
