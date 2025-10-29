"""Base model interface for OmniRay."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, **kwargs):
        """Initialize model with optional kwargs."""
        self.model_kwargs = kwargs

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Run inference on a single frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in RGB format
            **kwargs: Additional prediction arguments

        Returns:
            Dictionary containing prediction results
        """
        pass

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame before inference.

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        return frame

    def postprocess(self, outputs: Any) -> Dict[str, Any]:
        """Postprocess model outputs.

        Args:
            outputs: Raw model outputs

        Returns:
            Processed results as dictionary
        """
        return {"outputs": outputs}

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of frames using Ray.

        Args:
            batch: Batch dictionary containing 'frame' key

        Returns:
            Batch dictionary with added 'predictions' key
        """
        frame = batch["frame"]
        preprocessed = self.preprocess(frame)
        predictions = self.predict(preprocessed)

        return {
            **batch,
            "predictions": predictions,
        }
