# OmniRay Examples

This directory contains examples demonstrating different use cases of OmniRay.

## Examples Overview

### 1. Object Detection
**File**: [example_object_detection.py](example_object_detection.py)

Uses YOLOv8 for object detection on video frames.

```bash
python examples/example_object_detection.py
```

### 2. Emotion Analysis
**File**: [example_emotion_analysis.py](example_emotion_analysis.py)

Uses py-feat for facial emotion analysis on video.

```bash
python examples/example_emotion_analysis.py
```

### 3. Custom Model
**File**: [example_custom_model.py](example_custom_model.py)

Shows how to integrate your own custom model.

```bash
python examples/example_custom_model.py
```

### 4. Configuration-based Execution
**File**: [run_from_config.py](run_from_config.py)

Run inference using YAML configuration files.

```bash
# Object detection
python examples/run_from_config.py examples/config_object_detection.yaml

# Custom model
python examples/run_from_config.py examples/config_custom_model.yaml
```

## Configuration Files

### Object Detection Config
**File**: [config_object_detection.yaml](config_object_detection.yaml)

Example configuration for YOLOv8 object detection.

### Custom Model Config
**File**: [config_custom_model.yaml](config_custom_model.yaml)

Example configuration for custom models with preprocessing and postprocessing.

## Before Running

1. **Install dependencies**:
   ```bash
   pip install -e .
   # or
   uv pip install -e .
   ```

2. **Update video paths** in the example files or config files:
   ```python
   video_path: "path/to/your/video.mp4"
   ```

3. **For custom models**, implement your model class:
   ```python
   class MyCustomModel:
       def __init__(self, model_path=None, **kwargs):
           # Initialize your model
           pass

       def predict(self, frame, **kwargs):
           # Run inference
           return {"predictions": ...}
   ```

## Output

Results are saved to the `results/` directory by default:
- JSON format: Human-readable, good for small datasets
- Parquet format: Efficient, good for large datasets

## GPU Usage

To use GPU for inference, update the `ray_options` in config:

```yaml
ray_options:
  num_gpus: 1  # Use 1 GPU
  num_cpus: 4
```

Or in Python:

```python
config = InferenceConfig(
    model_type=ModelType.OBJECT_DETECTION,
    video_config=VideoConfig(video_path="video.mp4"),
    ray_options={"num_gpus": 1, "num_cpus": 4}
)
```

## Tips

- **Frame skipping**: Use `frame_skip` to process every Nth frame for faster processing
- **Batch size**: Adjust `batch_size` based on your memory constraints
- **Max frames**: Set `max_frames` for quick testing on a subset of video
- **Target size**: Resize frames using `target_size` to reduce computation

Example for fast prototyping:

```python
video_config = VideoConfig(
    video_path="video.mp4",
    frame_skip=10,      # Process every 10th frame
    max_frames=100,     # Only process 100 frames
    target_size=(320, 240)  # Resize to smaller resolution
)
```
