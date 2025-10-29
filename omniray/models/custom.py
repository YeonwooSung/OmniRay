"""Custom model loader from configuration."""

import importlib
import logging
from typing import Any, Dict

import numpy as np

from omniray.config.schemas import CustomModelConfig
from omniray.models.base import BaseModel

logger = logging.getLogger(__name__)


class CustomModel(BaseModel):
    """Custom model loader from YAML configuration."""

    def __init__(self, config: CustomModelConfig, **kwargs):
        """Initialize custom model.

        Args:
            config: Custom model configuration
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.config = config
        self.model = None
        self.model_class = None

    def _load_model_class(self):
        """Dynamically load model class from string."""
        try:
            module_path, class_name = self.config.model_class.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.model_class = getattr(module, class_name)
            logger.info(f"Loaded model class: {self.config.model_class}")
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import model class '{self.config.model_class}': {e}\n"
                f"Make sure the module is installed and the class path is correct."
            )

    def load_model(self) -> None:
        """Load custom model."""
        logger.info(f"Loading custom model: {self.config.model_class}")

        # Load the model class
        self._load_model_class()

        # Initialize model with config kwargs
        model_kwargs = self.config.model_kwargs.copy()

        # Add model path if provided
        if self.config.model_path:
            model_kwargs["model_path"] = self.config.model_path

        # Instantiate model
        self.model = self.model_class(**model_kwargs)

        # If model has a load method, call it
        if hasattr(self.model, "load"):
            self.model.load()

        logger.info("Custom model loaded successfully")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame using custom configuration.

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        if self.config.preprocessing is None:
            return frame

        # Apply custom preprocessing if defined
        # This is a simple implementation - can be extended
        preprocessed = frame

        if "resize" in self.config.preprocessing:
            import cv2

            size = self.config.preprocessing["resize"]
            preprocessed = cv2.resize(preprocessed, tuple(size))

        if "normalize" in self.config.preprocessing:
            mean = self.config.preprocessing["normalize"].get("mean", [0.0, 0.0, 0.0])
            std = self.config.preprocessing["normalize"].get("std", [1.0, 1.0, 1.0])
            preprocessed = (preprocessed - np.array(mean)) / np.array(std)

        return preprocessed

    def postprocess(self, outputs: Any) -> Dict[str, Any]:
        """Postprocess model outputs using custom configuration.

        Args:
            outputs: Raw model outputs

        Returns:
            Processed results as dictionary
        """
        if self.config.postprocessing is None:
            return {"outputs": outputs}

        # Apply custom postprocessing if defined
        # This is a simple implementation - can be extended
        processed = {"outputs": outputs}

        if "threshold" in self.config.postprocessing:
            threshold = self.config.postprocessing["threshold"]
            if isinstance(outputs, np.ndarray):
                processed["thresholded"] = (outputs > threshold).astype(int)

        return processed

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Run inference using custom model.

        Args:
            frame: Input frame as numpy array (H, W, C) in RGB format
            **kwargs: Additional prediction arguments

        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            self.load_model()

        # Call model's predict method or __call__
        if hasattr(self.model, "predict"):
            outputs = self.model.predict(frame, **kwargs)
        elif callable(self.model):
            outputs = self.model(frame, **kwargs)
        else:
            raise AttributeError(
                f"Model class {self.config.model_class} must have a 'predict' method or be callable"
            )

        # Postprocess outputs
        return self.postprocess(outputs)
