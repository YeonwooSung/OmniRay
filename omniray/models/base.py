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
            batch: Batch dictionary containing 'frame' key with shape (batch_size, H, W, C)
                   Note: Ray Data may pass frames as various formats (numpy, pandas Series, etc.)

        Returns:
            Batch dictionary with added 'predictions' key
        """
        frames = batch["frame"]
        
        # Convert frames to proper numpy array format
        # Ray Data may pass data as pandas Series, list of arrays, or nested structures
        frames = self._ensure_numpy_batch(frames)
        
        preprocessed = self.preprocess(frames)
        predictions = self.predict(preprocessed)

        # Handle batch predictions - predictions should be a list of dicts
        # when processing a batch of frames
        if isinstance(predictions, list):
            # Predictions is already a list of results (one per frame)
            return {
                **batch,
                "predictions": predictions,
            }
        else:
            # Single prediction dict - wrap in list to match batch structure
            return {
                **batch,
                "predictions": [predictions],
            }

    def _ensure_numpy_batch(self, frames) -> np.ndarray:
        """Ensure frames are in proper numpy array format.
        
        Ray Data can pass batches in various formats:
        - numpy ndarray (ideal)
        - pandas Series of numpy arrays
        - list of numpy arrays
        - nested object arrays
        
        Args:
            frames: Input frames in various possible formats
            
        Returns:
            numpy array with shape (batch_size, H, W, C)
        """
        # Already a proper numpy array with correct dimensions
        if isinstance(frames, np.ndarray):
            if frames.dtype == object:
                # Object array - need to stack individual frames
                frame_list = [np.asarray(f) for f in frames]
                return np.stack(frame_list, axis=0)
            elif len(frames.shape) == 4:
                # Already (B, H, W, C) format
                return frames
            elif len(frames.shape) == 3:
                # Single frame (H, W, C) - add batch dimension
                return np.expand_dims(frames, axis=0)
            else:
                return frames
        
        # Pandas Series or similar iterable
        if hasattr(frames, 'values'):
            frames = frames.values
        
        # List or other iterable of frames
        if hasattr(frames, '__iter__') and not isinstance(frames, np.ndarray):
            frame_list = [np.asarray(f) for f in frames]
            if len(frame_list) > 0:
                return np.stack(frame_list, axis=0)
            return np.array([])
        
        # Fallback - try to convert directly
        return np.asarray(frames)
