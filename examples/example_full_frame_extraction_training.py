"""Example: Full frame extraction and training pipeline.

This example extracts ALL frames from videos (no skipping), saves them as images,
and then trains a model on the extracted dataset.

Usage:
    python example_full_frame_extraction_training.py config_full_frame_extraction_training.yaml
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import yaml
import ray

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omniray.labeling.emotion_labeler import EmotionPseudoLabeler
from omniray.labeling.label_storage import LabelFormat, LabelStorage
from omniray.labeling.frame_extractor import FrameExtractor
from omniray.training.extracted_dataset import ExtractedFrameDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def run_pseudo_labeling_and_extraction(config: dict) -> dict:
    """Run pseudo labeling and frame extraction.

    Args:
        config: Configuration dictionary

    Returns:
        Extraction statistics
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Pseudo Labeling & Frame Extraction")
    logger.info("=" * 60)

    video_config = config['video_config']
    labeling_config = config['pseudo_labeling']
    
    # Initialize labeler
    labeler = EmotionPseudoLabeler(
        detector=labeling_config['detector'],
        au_model=labeling_config['au_model'],
        emotion_model=labeling_config['emotion_model'],
        confidence_threshold=labeling_config['confidence_threshold'],
        num_gpus=config['ray']['num_gpus'],
        num_cpus=config['ray']['num_cpus'],
    )

    # Process each video
    all_extraction_stats = []
    
    for video_path in video_config['video_paths']:
        logger.info(f"\nProcessing video: {video_path}")
        
        # Generate labels
        video_name = Path(video_path).stem
        video_output_dir = Path(labeling_config['output_dir']) / video_name
        
        label_stats = labeler.generate_labels(
            video_path=video_path,
            output_dir=str(video_output_dir),
            label_format=LabelFormat(labeling_config['label_format']),
            batch_size=video_config['batch_size'],
            frame_skip=video_config['frame_skip'],
            max_frames=video_config.get('max_frames'),
            save_frames=False,  # We'll handle frame saving separately
        )
        
        logger.info(f"Generated {label_stats['total_labels']} labels")
        logger.info(f"Emotion distribution: {label_stats['emotion_distribution']}")
        
        # Load labels for extraction
        storage = LabelStorage(output_dir=str(video_output_dir))
        labels = storage.load_labels(label_stats['label_path'])
        
        # Extract frames if enabled
        if labeling_config.get('save_frames', True):
            frame_extraction_config = labeling_config.get('frame_extraction', {})
            
            # Initialize frame extractor
            extractor = FrameExtractor(
                output_dir=str(video_output_dir),
                save_format=frame_extraction_config.get('save_format', 'jpg'),
                jpeg_quality=frame_extraction_config.get('jpeg_quality', 95),
                organize_by_emotion=frame_extraction_config.get('organize_by_emotion', True),
                include_metadata=frame_extraction_config.get('include_metadata', True),
            )
            
            # Extract and save frames
            extraction_stats = extractor.extract_and_save_frames(
                video_path=video_path,
                labels=labels,
                save_full_frames=frame_extraction_config.get('save_full_frames', False),
            )
            
            # Create dataset index
            index_path = video_output_dir / "dataset_index.json"
            extractor.create_dataset_index(save_path=str(index_path))
            
            extraction_stats.update({
                'video_path': video_path,
                'label_path': label_stats['label_path'],
                'frames_directory': str(video_output_dir / 'frames'),
                'index_path': str(index_path),
            })
            
            all_extraction_stats.append(extraction_stats)
            
            logger.info(f"Extracted {extraction_stats['total_frames_saved']} frames")
    
    # Merge statistics
    total_frames = sum(s['total_frames_saved'] for s in all_extraction_stats)
    combined_emotion_dist = {}
    for stats in all_extraction_stats:
        for emotion, count in stats['frames_by_emotion'].items():
            combined_emotion_dist[emotion] = combined_emotion_dist.get(emotion, 0) + count
    
    logger.info("\n" + "=" * 60)
    logger.info("Extraction Summary")
    logger.info("=" * 60)
    logger.info(f"Total frames extracted: {total_frames}")
    logger.info(f"Combined emotion distribution: {combined_emotion_dist}")
    
    return {
        'total_frames': total_frames,
        'emotion_distribution': combined_emotion_dist,
        'per_video_stats': all_extraction_stats,
    }


def prepare_training_datasets(config: dict, extraction_stats: dict):
    """Prepare training datasets from extracted frames.

    Args:
        config: Configuration dictionary
        extraction_stats: Extraction statistics

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Preparing Training Datasets")
    logger.info("=" * 60)

    dataset_config = config['dataset']
    labeling_config = config['pseudo_labeling']
    
    # Collect all frames directories
    frames_dirs = []
    metadata_dirs = []
    
    for video_stats in extraction_stats['per_video_stats']:
        frames_dir = Path(video_stats['frames_directory'])
        metadata_dir = frames_dir.parent / 'metadata'
        
        if frames_dir.exists():
            frames_dirs.append(str(frames_dir))
            if metadata_dir.exists():
                metadata_dirs.append(str(metadata_dir))
    
    if not frames_dirs:
        raise ValueError("No extracted frames found!")
    
    # Create datasets from extracted frames
    try:
        import torch
        from torch.utils.data import ConcatDataset, random_split
    except ImportError:
        logger.error("PyTorch is required for training. Please install: pip install torch")
        return None, None, None
    
    # Load all frame datasets
    all_datasets = []
    for frames_dir, metadata_dir in zip(frames_dirs, metadata_dirs):
        dataset = ExtractedFrameDataset(
            frames_dir=frames_dir,
            metadata_dir=metadata_dir if metadata_dir else None,
            transform=None,  # Add transforms based on config
            target_size=tuple(dataset_config['target_size']),
            min_confidence=dataset_config.get('min_confidence'),
            image_format=labeling_config.get('frame_extraction', {}).get('save_format', 'jpg'),
        )
        all_datasets.append(dataset)
    
    # Concatenate all datasets
    full_dataset = ConcatDataset(all_datasets)
    
    # Split dataset
    split_ratio = dataset_config['split_ratio']
    total_size = len(full_dataset)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
    )
    
    logger.info(f"\nDataset Split:")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")
    
    # Show class distribution
    if all_datasets:
        dist = all_datasets[0].get_class_distribution()
        logger.info(f"\nClass Distribution: {dist}")
    
    return train_dataset, val_dataset, test_dataset


def run_training(config: dict, train_dataset, val_dataset, test_dataset):
    """Run model training.

    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset

    Returns:
        Training results
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Training Model")
    logger.info("=" * 60)

    try:
        import torch
        import torch.nn as nn
        from omniray.training.trainer import EmotionTrainingPipeline
    except ImportError as e:
        logger.error(f"Training dependencies not available: {e}")
        return None

    # Create model based on config
    model_config = config['model']
    architecture = model_config.get('architecture', 'simple_cnn')
    num_classes = model_config['num_classes']
    
    if architecture == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(pretrained=model_config.get('pretrained', False))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(pretrained=model_config.get('pretrained', False))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # Simple CNN
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, num_classes),
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = SimpleCNN(num_classes=num_classes)
    
    logger.info(f"Using model architecture: {architecture}")
    
    # Training configuration
    training_config = config['training']
    ray_config = config['ray']
    output_config = config['output']
    
    # Initialize training pipeline
    pipeline = EmotionTrainingPipeline(
        model=model,
        num_workers=ray_config['num_workers'],
        use_gpu=ray_config['use_gpu'],
        results_dir=output_config['results_dir'],
    )
    
    # Train
    results = pipeline.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        checkpoint_freq=training_config['checkpoint_freq'],
    )
    
    logger.info(f"\nTraining Status: {results['status']}")
    
    # Evaluate
    if results['status'] == 'completed':
        logger.info("\nEvaluating on test set...")
        eval_results = pipeline.evaluate(
            model=model,
            test_dataset=test_dataset,
            checkpoint_path=results.get('best_checkpoint'),
        )
        
        logger.info(f"\nTest Results:")
        logger.info(f"  Accuracy: {eval_results['test_accuracy']:.2f}%")
        logger.info(f"  Loss: {eval_results['test_loss']:.4f}")
        logger.info(f"  Per-class accuracies: {eval_results['class_accuracies']}")
        
        results['evaluation'] = eval_results
    
    return results


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Full frame extraction and training pipeline"
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip frame extraction (use existing frames)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Only extract frames, skip training'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Initialize Ray
    ray_config = config.get('ray', {})
    if not ray.is_initialized():
        ray.init(
            num_cpus=ray_config.get('num_cpus', 4),
            num_gpus=ray_config.get('num_gpus', 0),
            ignore_reinit_error=True,
        )
    
    try:
        extraction_stats = None
        
        # Phase 1: Pseudo labeling and frame extraction
        pipeline_config = config.get('pipeline', {})
        if not args.skip_extraction and pipeline_config.get('run_pseudo_labeling', True):
            extraction_stats = run_pseudo_labeling_and_extraction(config)
        else:
            logger.info("Skipping frame extraction, using existing frames")
            # Load existing stats if available
            # TODO: Implement loading existing extraction stats
        
        # Phase 2 & 3: Training
        if not args.skip_training and pipeline_config.get('run_training', True):
            if extraction_stats:
                train_ds, val_ds, test_ds = prepare_training_datasets(config, extraction_stats)
                
                if train_ds and val_ds and test_ds:
                    training_results = run_training(config, train_ds, val_ds, test_ds)
                    
                    if training_results:
                        logger.info("\n" + "=" * 60)
                        logger.info("Pipeline Completed Successfully!")
                        logger.info("=" * 60)
            else:
                logger.warning("No extraction stats available, cannot prepare datasets")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
    
    finally:
        ray.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
