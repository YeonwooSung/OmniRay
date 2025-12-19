"""Example: Run pseudo labeling and training from YAML config file.

Usage:
    python example_pseudo_labeling_from_config.py config_pseudo_labeling_training.yaml
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
from omniray.labeling.label_storage import LabelFormat
from omniray.training import DatasetBuilder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def run_pseudo_labeling(config: dict) -> list:
    """Run pseudo labeling based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of label statistics
    """
    logger.info("=" * 60)
    logger.info("Starting Pseudo Labeling Phase")
    logger.info("=" * 60)

    # Extract configuration
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

    # Generate labels
    label_format = LabelFormat(labeling_config['label_format'])
    
    all_stats = labeler.generate_labels_batch(
        video_paths=video_config['video_paths'],
        output_dir=labeling_config['output_dir'],
        label_format=label_format,
        batch_size=video_config['batch_size'],
        frame_skip=video_config['frame_skip'],
        max_frames=video_config.get('max_frames'),
        save_frames=labeling_config.get('save_frames', False),
    )

    # Log results
    logger.info("\nPseudo Labeling Results:")
    for stats in all_stats:
        if 'error' not in stats:
            logger.info(f"\nVideo: {Path(stats['video_path']).name}")
            logger.info(f"  Labels generated: {stats['total_labels']}")
            logger.info(f"  Unique frames: {stats.get('unique_frames', 0)}")
            logger.info(f"  Emotion distribution: {stats.get('emotion_distribution', {})}")
            avg_conf = stats.get('avg_confidence', 0.0)
            logger.info(f"  Average confidence: {avg_conf:.3f}")
            
            if stats['total_labels'] == 0:
                logger.warning("  ⚠️ No faces detected in this video. Check if:")
                logger.warning("    - Video contains visible faces")
                logger.warning("    - Faces are large enough to detect")
                logger.warning("    - Video format is supported")
        else:
            logger.error(f"\nFailed to process: {stats['video_path']}")
            logger.error(f"  Error: {stats['error']}")

    return all_stats


def prepare_training_data(config: dict, label_stats: list):
    """Prepare training datasets from pseudo labels.

    Args:
        config: Configuration dictionary
        label_stats: Label statistics from pseudo labeling

    Returns:
        Training datasets tuple
    """
    logger.info("=" * 60)
    logger.info("Preparing Training Data")
    logger.info("=" * 60)

    # Filter successful labelings
    label_video_pairs = [
        (stat['label_path'], stat['video_path'])
        for stat in label_stats
        if 'error' not in stat
    ]

    if not label_video_pairs:
        raise ValueError("No valid labels generated. Cannot proceed with training.")

    # Build datasets
    dataset_config = config['dataset']
    builder = DatasetBuilder()

    # Note: In production, you would implement proper transforms here
    train_dataset, val_dataset, test_dataset = builder.build_datasets_from_multiple_videos(
        label_video_pairs=label_video_pairs,
        transform=None,  # Add transforms based on config if needed
        split_ratio=tuple(dataset_config['split_ratio']),
        min_confidence=dataset_config['min_confidence'],
        target_size=tuple(dataset_config['target_size']),
    )

    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")
    logger.info(f"  Test samples: {len(test_dataset)}")

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
    logger.info("Starting Model Training")
    logger.info("=" * 60)

    # Import here to avoid errors if torch is not installed
    try:
        import torch.nn as nn
        from omniray.training.trainer import EmotionTrainingPipeline
    except ImportError as e:
        logger.error(f"Training dependencies not available: {e}")
        logger.error("Please install PyTorch and Ray Train to use training features")
        return None

    # Create simple model (you can customize this)
    class SimpleModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, num_classes),
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    # Extract training configuration
    model_config = config['model']
    training_config = config['training']
    ray_config = config['ray']
    output_config = config['output']

    # Create model
    model = SimpleModel(num_classes=model_config['num_classes'])

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
    
    # Evaluate if training succeeded
    if results['status'] == 'completed':
        logger.info("\nEvaluating on test set...")
        eval_results = pipeline.evaluate(
            model=model,
            test_dataset=test_dataset,
            checkpoint_path=results.get('best_checkpoint'),
        )
        
        logger.info(f"\nTest Results:")
        logger.info(f"  Test Accuracy: {eval_results['test_accuracy']:.2f}%")
        logger.info(f"  Test Loss: {eval_results['test_loss']:.4f}")
        
        results['evaluation'] = eval_results

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run pseudo labeling and training from config file"
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--labeling-only',
        action='store_true',
        help='Only run pseudo labeling, skip training'
    )
    parser.add_argument(
        '--training-only',
        action='store_true',
        help='Only run training (assumes labels exist)'
    )

    args = parser.parse_args()

    # Load configuration
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
        # Run pseudo labeling
        if not args.training_only:
            label_stats = run_pseudo_labeling(config)
        else:
            # Load existing labels (implement as needed)
            logger.info("Skipping pseudo labeling, using existing labels")
            label_stats = []  # Load from config or directory

        # Run training
        if not args.labeling_only and label_stats:
            train_dataset, val_dataset, test_dataset = prepare_training_data(
                config, label_stats
            )

            training_results = run_training(
                config, train_dataset, val_dataset, test_dataset
            )

            if training_results:
                logger.info("\n" + "=" * 60)
                logger.info("Pipeline Completed Successfully!")
                logger.info("=" * 60)

        elif args.labeling_only:
            logger.info("\nPseudo labeling completed. Skipping training.")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
