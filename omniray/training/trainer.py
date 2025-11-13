"""Ray-based distributed training pipeline for emotion recognition."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EmotionTrainingPipeline:
    """Ray-based distributed training pipeline for emotion recognition.
    
    This pipeline uses Ray Train to distribute training across multiple workers,
    enabling efficient training on pseudo-labeled emotion data.
    """

    def __init__(
        self,
        model: nn.Module,
        num_workers: int = 2,
        use_gpu: bool = True,
        results_dir: str = "./training_results",
    ):
        """Initialize training pipeline.

        Args:
            model: PyTorch model to train
            num_workers: Number of Ray training workers
            use_gpu: Whether to use GPU for training
            results_dir: Directory to save training results
        """
        self.model = model
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_freq: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run distributed training.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            batch_size: Batch size per worker
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            checkpoint_freq: Checkpoint frequency (epochs)
            **kwargs: Additional training arguments

        Returns:
            Training results dictionary
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create training function
        def train_func(config: Dict[str, Any]):
            """Training function to run on each worker."""
            # Get distributed training context
            world_size = train.get_context().get_world_size()
            world_rank = train.get_context().get_world_rank()
            
            logger.info(f"Worker {world_rank}/{world_size} starting training")

            # Prepare model
            model = config["model"]
            device = torch.device("cuda" if torch.cuda.is_available() and config["use_gpu"] else "cpu")
            model = model.to(device)

            # Wrap model for distributed training
            model = train.torch.prepare_model(model)

            # Prepare data loaders
            train_loader = DataLoader(
                config["train_dataset"],
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=2,
            )
            train_loader = train.torch.prepare_data_loader(train_loader)

            val_loader = None
            if config.get("val_dataset") is not None:
                val_loader = DataLoader(
                    config["val_dataset"],
                    batch_size=config["batch_size"],
                    shuffle=False,
                    num_workers=2,
                )
                val_loader = train.torch.prepare_data_loader(val_loader)

            # Setup optimizer and loss
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
            criterion = nn.CrossEntropyLoss()

            # Training loop
            for epoch in range(config["epochs"]):
                # Train
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_idx, (data, target, _) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()

                avg_train_loss = train_loss / len(train_loader)
                train_acc = 100.0 * train_correct / train_total

                # Validation
                val_loss = 0.0
                val_acc = 0.0
                if val_loader is not None:
                    model.eval()
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for data, target, _ in val_loader:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            loss = criterion(output, target)

                            val_loss += loss.item()
                            _, predicted = output.max(1)
                            val_total += target.size(0)
                            val_correct += predicted.eq(target).sum().item()

                    val_loss = val_loss / len(val_loader)
                    val_acc = 100.0 * val_correct / val_total

                # Report metrics
                metrics = {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }

                # Save checkpoint
                if (epoch + 1) % config["checkpoint_freq"] == 0:
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    train.report(metrics, checkpoint=checkpoint)
                else:
                    train.report(metrics)

                logger.info(
                    f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                    f"train_acc={train_acc:.2f}%, val_loss={val_loss:.4f}, "
                    f"val_acc={val_acc:.2f}%"
                )

        # Configure scaling
        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu,
            resources_per_worker={"CPU": 2, "GPU": 1 if self.use_gpu else 0},
        )

        # Configure checkpointing
        checkpoint_config = CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="val_acc",
            checkpoint_score_order="max",
        )

        # Configure run
        run_config = RunConfig(
            name="emotion_training",
            storage_path=str(self.results_dir),
            checkpoint_config=checkpoint_config,
        )

        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config={
                "model": self.model,
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "use_gpu": self.use_gpu,
                "checkpoint_freq": checkpoint_freq,
                **kwargs,
            },
            scaling_config=scaling_config,
            run_config=run_config,
        )

        # Run training
        logger.info("Starting distributed training...")
        result = trainer.fit()

        # Get best checkpoint
        best_checkpoint = result.best_checkpoints[0][0] if result.best_checkpoints else None

        training_results = {
            "status": "completed" if result.error is None else "failed",
            "best_checkpoint": best_checkpoint,
            "metrics": result.metrics,
            "error": str(result.error) if result.error else None,
        }

        logger.info(f"Training completed. Results: {training_results}")
        return training_results

    def evaluate(
        self,
        model: nn.Module,
        test_dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate model on test set.

        Args:
            model: Model to evaluate
            test_dataset: Test dataset
            batch_size: Batch size for evaluation
            checkpoint_path: Optional checkpoint to load

        Returns:
            Evaluation results
        """
        device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")
        model = model.to(device)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

        model.eval()

        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0

        # Per-class accuracy
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for data, target, metadata in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Track per-class accuracy
                for t, p in zip(target, predicted):
                    t_item = t.item()
                    if t_item not in class_correct:
                        class_correct[t_item] = 0
                        class_total[t_item] = 0
                    class_total[t_item] += 1
                    if t == p:
                        class_correct[t_item] += 1

        avg_loss = test_loss / len(test_loader)
        accuracy = 100.0 * correct / total

        # Calculate per-class accuracy
        class_accuracies = {
            cls: 100.0 * class_correct[cls] / class_total[cls]
            for cls in class_correct.keys()
        }

        results = {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "class_accuracies": class_accuracies,
        }

        logger.info(
            f"Evaluation results: loss={avg_loss:.4f}, accuracy={accuracy:.2f}%"
        )

        return results
