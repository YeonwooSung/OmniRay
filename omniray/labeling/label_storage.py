"""Label storage utilities for pseudo labels."""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class LabelFormat(str, Enum):
    """Supported label storage formats."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


class LabelStorage:
    """Storage handler for pseudo labels."""

    def __init__(self, output_dir: str):
        """Initialize label storage.

        Args:
            output_dir: Directory for storing labels
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_labels(
        self,
        labels: List[Dict[str, Any]],
        format: LabelFormat = LabelFormat.JSON,
        filename_prefix: str = "labels",
    ) -> Path:
        """Save labels to file.

        Args:
            labels: List of label dictionaries
            format: Output format (JSON, CSV, PARQUET)
            filename_prefix: Prefix for output filename

        Returns:
            Path to saved file
        """
        if format == LabelFormat.JSON:
            return self._save_json(labels, filename_prefix)
        elif format == LabelFormat.CSV:
            return self._save_csv(labels, filename_prefix)
        elif format == LabelFormat.PARQUET:
            return self._save_parquet(labels, filename_prefix)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_json(self, labels: List[Dict[str, Any]], prefix: str) -> Path:
        """Save labels as JSON."""
        output_path = self.output_dir / f"{prefix}_labels.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(labels)} labels to {output_path}")
        return output_path

    def _save_csv(self, labels: List[Dict[str, Any]], prefix: str) -> Path:
        """Save labels as CSV."""
        output_path = self.output_dir / f"{prefix}_labels.csv"
        
        # Flatten nested dictionaries for CSV
        flattened_labels = []
        for label in labels:
            flat_label = self._flatten_dict(label)
            flattened_labels.append(flat_label)
        
        df = pd.DataFrame(flattened_labels)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(labels)} labels to {output_path}")
        return output_path

    def _save_parquet(self, labels: List[Dict[str, Any]], prefix: str) -> Path:
        """Save labels as Parquet."""
        output_path = self.output_dir / f"{prefix}_labels.parquet"
        
        df = pd.DataFrame(labels)
        df.to_parquet(output_path, index=False)
        
        logger.info(f"Saved {len(labels)} labels to {output_path}")
        return output_path

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "_"
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for CSV storage.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings for CSV
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)

    def load_labels(self, filepath: str) -> List[Dict[str, Any]]:
        """Load labels from file.

        Args:
            filepath: Path to label file

        Returns:
            List of label dictionaries
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Label file not found: {filepath}")

        if filepath.suffix == ".json":
            return self._load_json(filepath)
        elif filepath.suffix == ".csv":
            return self._load_csv(filepath)
        elif filepath.suffix == ".parquet":
            return self._load_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

    def _load_json(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load labels from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        logger.info(f"Loaded {len(labels)} labels from {filepath}")
        return labels

    def _load_csv(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load labels from CSV."""
        df = pd.read_csv(filepath)
        labels = df.to_dict("records")
        
        logger.info(f"Loaded {len(labels)} labels from {filepath}")
        return labels

    def _load_parquet(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load labels from Parquet."""
        df = pd.read_parquet(filepath)
        labels = df.to_dict("records")
        
        logger.info(f"Loaded {len(labels)} labels from {filepath}")
        return labels

    def merge_labels(
        self,
        label_files: List[str],
        output_path: str,
        format: LabelFormat = LabelFormat.JSON,
    ) -> Path:
        """Merge multiple label files into one.

        Args:
            label_files: List of label file paths
            output_path: Output path for merged labels
            format: Output format

        Returns:
            Path to merged file
        """
        all_labels = []
        
        for label_file in label_files:
            labels = self.load_labels(label_file)
            all_labels.extend(labels)
        
        logger.info(f"Merged {len(all_labels)} labels from {len(label_files)} files")
        
        # Save merged labels
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == LabelFormat.JSON:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_labels, f, indent=2, ensure_ascii=False)
        elif format == LabelFormat.CSV:
            flattened = [self._flatten_dict(label) for label in all_labels]
            df = pd.DataFrame(flattened)
            df.to_csv(output_path, index=False)
        elif format == LabelFormat.PARQUET:
            df = pd.DataFrame(all_labels)
            df.to_parquet(output_path, index=False)
        
        logger.info(f"Saved merged labels to {output_path}")
        return output_path

    def filter_labels(
        self,
        labels: List[Dict[str, Any]],
        min_confidence: Optional[float] = None,
        emotions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter labels by criteria.

        Args:
            labels: List of label dictionaries
            min_confidence: Minimum confidence threshold
            emotions: List of emotions to keep

        Returns:
            Filtered list of labels
        """
        filtered = labels

        if min_confidence is not None:
            filtered = [
                label for label in filtered
                if label.get("emotion_confidence", 0) >= min_confidence
            ]
            logger.info(f"Filtered by confidence >= {min_confidence}: {len(filtered)} labels")

        if emotions is not None:
            filtered = [
                label for label in filtered
                if label.get("dominant_emotion") in emotions
            ]
            logger.info(f"Filtered by emotions {emotions}: {len(filtered)} labels")

        return filtered
