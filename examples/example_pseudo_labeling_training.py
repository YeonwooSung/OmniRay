"""Example: Generate pseudo labels and train emotion model.

This example demonstrates the full pipeline:
1. Generate pseudo labels from videos using emotion analysis
2. Prepare training dataset from pseudo labels
3. Train emotion recognition model using Ray distributed training
"""

import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
import torch.nn as nn
from torchvision import transforms

from omniray.labeling.emotion_labeler import EmotionPseudoLabeler
from omniray.labeling.label_storage import LabelFormat
from omniray.training import DatasetBuilder
from omniray.training.trainer import EmotionTrainingPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleEmotionCNN(nn.Module):
    """Simple CNN for emotion recognition (for demonstration)."""

    def __init__(self, num_classes: int = 7):
        """Initialize CNN.

        Args:
            num_classes: Number of emotion classes
        """
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x


def generate_pseudo_labels(video_paths: list[str], output_dir: str):
    """Step 1: Generate pseudo labels from videos.

    Args:
        video_paths: List of video file paths
        output_dir: Output directory for labels
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Generating Pseudo Labels")
    logger.info("=" * 60)

    # Initialize pseudo labeler
    labeler = EmotionPseudoLabeler(
        detector="retinaface",
        emotion_model="resmasknet",
        confidence_threshold=0.6,  # Only keep high-confidence labels
        num_gpus=0,  # Set to 1 if you have GPU
        num_cpus=4,
    )

    # Generate labels for all videos
    all_stats = labeler.generate_labels_batch(
        video_paths=video_paths,
        output_dir=output_dir,
        label_format=LabelFormat.JSON,
        batch_size=16,
        frame_skip=5,  # Process every 5th frame
        max_frames=200,  # Limit for demonstration
    )

    # Print statistics
    for stats in all_stats:
        if "error" not in stats:
            logger.info(f"\nVideo: {stats['video_path']}")
            logger.info(f"  Total labels: {stats['total_labels']}")
            logger.info(f"  Emotion distribution: {stats['emotion_distribution']}")
            logger.info(f"  Avg confidence: {stats['avg_confidence']:.2f}")
            logger.info(f"  Labels saved to: {stats['label_path']}")

    return all_stats


def prepare_datasets(label_video_pairs: list[tuple[str, str]], min_confidence: float = 0.7):
    """Step 2: Prepare training datasets.

    Args:
        label_video_pairs: List of (label_path, video_path) tuples
        min_confidence: Minimum confidence threshold

    Returns:
        Train, validation, and test datasets
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Preparing Training Datasets")
    logger.info("=" * 60)

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Build datasets
    builder = DatasetBuilder()
    train_dataset, val_dataset, test_dataset = builder.build_datasets_from_multiple_videos(
        label_video_pairs=label_video_pairs,
        transform=transform,
        split_ratio=(0.7, 0.15, 0.15),
        min_confidence=min_confidence,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def train_model(
    train_dataset,
    val_dataset,
    test_dataset,
    num_classes: int = 7,
    epochs: int = 10,
    num_workers: int = 2,
):
    """Step 3: Train emotion recognition model.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        num_classes: Number of emotion classes
        epochs: Number of training epochs
        num_workers: Number of Ray workers

    Returns:
        Training results
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Training Emotion Recognition Model")
    logger.info("=" * 60)

    # Create model
    model = SimpleEmotionCNN(num_classes=num_classes)

    # Initialize Ray if not already
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create training pipeline
    pipeline = EmotionTrainingPipeline(
        model=model,
        num_workers=num_workers,
        use_gpu=False,  # Set to True if you have GPU
        results_dir="./emotion_training_results",
    )

    # Train
    training_results = pipeline.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=32,
        learning_rate=1e-3,
        checkpoint_freq=2,
    )

    logger.info(f"\nTraining status: {training_results['status']}")
    if training_results['best_checkpoint']:
        logger.info(f"Best checkpoint saved at: {training_results['best_checkpoint']}")

    # Evaluate on test set
    if training_results['status'] == 'completed':
        logger.info("\nEvaluating on test set...")
        eval_results = pipeline.evaluate(
            model=model,
            test_dataset=test_dataset,
            checkpoint_path=training_results['best_checkpoint'],
        )
        
        logger.info(f"\nTest Results:")
        logger.info(f"  Accuracy: {eval_results['test_accuracy']:.2f}%")
        logger.info(f"  Loss: {eval_results['test_loss']:.4f}")
        logger.info(f"  Per-class accuracies: {eval_results['class_accuracies']}")

    return training_results


def main():
    """Main pipeline execution."""
    # Configuration
    video_paths = [
        "path/to/video1.mp4",
        "path/to/video2.mp4",
    ]
    output_dir = "./pseudo_labels"

    # Step 1: Generate pseudo labels
    stats = generate_pseudo_labels(video_paths, output_dir)

    # Prepare label-video pairs
    label_video_pairs = [
        (stat['label_path'], stat['video_path'])
        for stat in stats
        if 'error' not in stat
    ]

    if not label_video_pairs:
        logger.error("No valid labels generated. Exiting.")
        return

    # Step 2: Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        label_video_pairs=label_video_pairs,
        min_confidence=0.7,
    )

    # Step 3: Train model
    training_results = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_classes=7,  # Adjust based on your emotions
        epochs=10,
        num_workers=2,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    # Get video paths from user
    print("Enter video paths (comma-separated):")
    video_input = input().strip()
    
    if video_input:
        video_paths = [path.strip() for path in video_input.split(",")]
        
        # Override configuration
        output_dir = "./pseudo_labels"
        
        # Run pipeline with user-provided videos
        stats = generate_pseudo_labels(video_paths, output_dir)
        
        label_video_pairs = [
            (stat['label_path'], stat['video_path'])
            for stat in stats
            if 'error' not in stat
        ]
        
        if label_video_pairs:
            train_dataset, val_dataset, test_dataset = prepare_datasets(
                label_video_pairs=label_video_pairs,
                min_confidence=0.7,
            )
            
            training_results = train_model(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                num_classes=7,
                epochs=10,
                num_workers=2,
            )
        
        ray.shutdown()
    else:
        # Run with default configuration
        main()
