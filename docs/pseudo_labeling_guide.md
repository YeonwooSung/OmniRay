# Pseudo Labeling and Training Guide

This guide explains how to use OmniRay's pseudo labeling and training pipeline for emotion recognition.

## Overview

The pseudo labeling and training pipeline consists of three main steps:

1. **Pseudo Label Generation**: Extract emotion labels from videos using pre-trained models
2. **Dataset Preparation**: Convert pseudo labels into training-ready datasets
3. **Distributed Training**: Train emotion recognition models using Ray distributed training

## Quick Start

### 1. Generate Pseudo Labels

```python
from omniray.labeling.emotion_labeler import EmotionPseudoLabeler
from omniray.labeling.label_storage import LabelFormat

# Initialize labeler
labeler = EmotionPseudoLabeler(
    detector="retinaface",
    emotion_model="resmasknet",
    confidence_threshold=0.6,
)

# Generate labels
stats = labeler.generate_labels(
    video_path="path/to/video.mp4",
    output_dir="./labels",
    label_format=LabelFormat.JSON,
    batch_size=16,
    frame_skip=5,
)

print(f"Generated {stats['total_labels']} labels")
print(f"Emotion distribution: {stats['emotion_distribution']}")
```

### 2. Prepare Training Dataset

```python
from omniray.training import DatasetBuilder

builder = DatasetBuilder()

# Build dataset from labels
dataset = builder.build_dataset(
    label_path="./labels/video_labels.json",
    video_path="path/to/video.mp4",
    min_confidence=0.7,
)

# Or build from multiple videos
train_ds, val_ds, test_ds = builder.build_datasets_from_multiple_videos(
    label_video_pairs=[
        ("./labels/video1_labels.json", "path/to/video1.mp4"),
        ("./labels/video2_labels.json", "path/to/video2.mp4"),
    ],
    split_ratio=(0.7, 0.15, 0.15),
)
```

### 3. Train Model with Ray

```python
import torch.nn as nn
from omniray.training.trainer import EmotionTrainingPipeline

# Define your model
model = YourEmotionModel(num_classes=7)

# Initialize training pipeline
pipeline = EmotionTrainingPipeline(
    model=model,
    num_workers=2,
    use_gpu=True,
)

# Train
results = pipeline.train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
)

# Evaluate
eval_results = pipeline.evaluate(
    model=model,
    test_dataset=test_ds,
    checkpoint_path=results['best_checkpoint'],
)
```

## Using Configuration Files

You can also run the entire pipeline using a YAML configuration file:

```bash
python examples/example_pseudo_labeling_from_config.py examples/config_pseudo_labeling_training.yaml
```

### Configuration File Structure

```yaml
video_config:
  video_paths:
    - "path/to/video1.mp4"
    - "path/to/video2.mp4"
  batch_size: 16
  frame_skip: 5

pseudo_labeling:
  detector: "retinaface"
  emotion_model: "resmasknet"
  confidence_threshold: 0.6
  output_dir: "./pseudo_labels"
  label_format: "json"

dataset:
  min_confidence: 0.7
  target_size: [224, 224]
  split_ratio: [0.7, 0.15, 0.15]

model:
  num_classes: 7

training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001

ray:
  num_workers: 2
  num_gpus: 1
```

## Advanced Features

### Label Storage and Management

```python
from omniray.labeling.label_storage import LabelStorage, LabelFormat

storage = LabelStorage(output_dir="./labels")

# Save labels in different formats
storage.save_labels(labels, format=LabelFormat.JSON)
storage.save_labels(labels, format=LabelFormat.CSV)
storage.save_labels(labels, format=LabelFormat.PARQUET)

# Load labels
labels = storage.load_labels("./labels/video_labels.json")

# Filter labels
filtered = storage.filter_labels(
    labels,
    min_confidence=0.8,
    emotions=["happiness", "surprise"],
)

# Merge multiple label files
merged_path = storage.merge_labels(
    label_files=["labels1.json", "labels2.json"],
    output_path="merged_labels.json",
)
```

### Batch Processing Multiple Videos

```python
labeler = EmotionPseudoLabeler()

# Process multiple videos
all_stats = labeler.generate_labels_batch(
    video_paths=[
        "video1.mp4",
        "video2.mp4",
        "video3.mp4",
    ],
    output_dir="./batch_labels",
    label_format=LabelFormat.JSON,
)

# Check results
for stats in all_stats:
    if 'error' in stats:
        print(f"Failed: {stats['video_path']}")
    else:
        print(f"Success: {stats['total_labels']} labels from {stats['video_path']}")
```

### Export for External Training

If you want to use the labeled data with other training frameworks:

```python
builder = DatasetBuilder()

# Export frames and labels
export_stats = builder.export_for_training(
    label_path="./labels/video_labels.json",
    video_path="path/to/video.mp4",
    output_dir="./exported_data",
    extract_frames=True,
    min_confidence=0.7,
)

# This creates a directory structure:
# exported_data/
#   frames/
#     happiness/
#       frame_0_face_0.jpg
#       frame_5_face_0.jpg
#     sadness/
#       frame_10_face_0.jpg
#   filtered_labels.json
```

## Emotion Classes

The default emotion classes used are:
- Anger
- Disgust
- Fear
- Happiness
- Sadness
- Surprise
- Neutral

## Performance Tips

1. **Frame Sampling**: Use `frame_skip` to process every Nth frame for faster processing
2. **Confidence Filtering**: Set higher `confidence_threshold` for cleaner pseudo labels
3. **Batch Size**: Adjust `batch_size` based on your GPU memory
4. **Distributed Training**: Use multiple workers with Ray for faster training

## Examples

See the `examples/` directory for complete examples:

- `example_pseudo_labeling_training.py`: Full pipeline example
- `example_pseudo_labeling_from_config.py`: Config-based execution
- `config_pseudo_labeling_training.yaml`: Example configuration file

## Requirements

For pseudo labeling:
```
ray[data]
py-feat
opencv-python
pandas
```

For training:
```
torch
torchvision
ray[train]
```

Install all dependencies:
```bash
pip install ray[data,train] py-feat opencv-python pandas torch torchvision
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Increase `frame_skip`
- Use `max_frames` to limit processing

### Low Quality Labels
- Increase `confidence_threshold`
- Use better quality input videos
- Try different detector models

### Slow Training
- Increase `num_workers`
- Enable GPU with `use_gpu=True`
- Use data augmentation wisely

## Next Steps

After training:
1. Evaluate model on held-out test set
2. Export trained model for deployment
3. Use trained model for inference on new videos
4. Iterate on pseudo labels to improve quality
