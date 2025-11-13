# OmniRay Examples

This directory contains examples demonstrating different use cases of OmniRay.

## Examples Overview

### Vision Processing

#### 1. Object Detection
**File**: [example_object_detection.py](example_object_detection.py)

Uses YOLOv8 for object detection on video frames.

```bash
python examples/example_object_detection.py
```

#### 2. Emotion Analysis
**File**: [example_emotion_analysis.py](example_emotion_analysis.py)

Uses py-feat for facial emotion analysis on video.

```bash
python examples/example_emotion_analysis.py
```

#### 3. Custom Model
**File**: [example_custom_model.py](example_custom_model.py)

Shows how to integrate your own custom model.

```bash
python examples/example_custom_model.py
```

### Pseudo Labeling & Training

#### 4. Pseudo Labeling with Training
**File**: [example_pseudo_labeling_training.py](example_pseudo_labeling_training.py)

Generate pseudo labels from videos and train emotion recognition models.

```bash
python examples/example_pseudo_labeling_training.py
```

#### 5. Pseudo Labeling from Config
**File**: [example_pseudo_labeling_from_config.py](example_pseudo_labeling_from_config.py)

Run pseudo labeling and training pipeline using YAML configuration.

```bash
python examples/example_pseudo_labeling_from_config.py examples/config_pseudo_labeling_training.yaml
```

#### 6. Full Frame Extraction & Training
**File**: [example_full_frame_extraction_training.py](example_full_frame_extraction_training.py)

Extract ALL frames from videos, save as images, and train on the full dataset.

```bash
python examples/example_full_frame_extraction_training.py examples/config_full_frame_extraction_training.yaml

# Only extract frames
python examples/example_full_frame_extraction_training.py examples/config_full_frame_extraction_training.yaml --skip-training

# Only train (use existing frames)
python examples/example_full_frame_extraction_training.py examples/config_full_frame_extraction_training.yaml --skip-extraction
```

### Configuration-based Execution

#### 7. Run from Config
**File**: [run_from_config.py](run_from_config.py)

Run inference using YAML configuration files.

```bash
# Object detection
python examples/run_from_config.py examples/config_object_detection.yaml

# Custom model
python examples/run_from_config.py examples/config_custom_model.yaml
```

## Configuration Files

### Vision Processing Configs

#### Object Detection Config
**File**: [config_object_detection.yaml](config_object_detection.yaml)

Example configuration for YOLOv8 object detection.

### Custom Model Config
**File**: [config_custom_model.yaml](config_custom_model.yaml)

Example configuration for custom models with preprocessing and postprocessing.

### Training & Labeling Configs

#### Pseudo Labeling Training Config
**File**: [config_pseudo_labeling_training.yaml](config_pseudo_labeling_training.yaml)

Configuration for generating pseudo labels and training (with frame skipping for faster processing).

```yaml
video_config:
  frame_skip: 5  # Process every 5th frame
  max_frames: 500
pseudo_labeling:
  save_frames: false  # Labels only
```

#### Full Frame Extraction Config
**File**: [config_full_frame_extraction_training.yaml](config_full_frame_extraction_training.yaml)

Configuration for extracting ALL frames and training on the complete dataset.

```yaml
video_config:
  frame_skip: 1  # Process EVERY frame
  max_frames: null  # All frames
pseudo_labeling:
  save_frames: true  # Save all frames as images
  frame_extraction:
    organize_by_emotion: true
    include_metadata: true
```

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
