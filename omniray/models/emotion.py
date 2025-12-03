"""Emotion analysis model using py-feat."""

import logging
from typing import Any, Dict, List, Union

import numpy as np

from omniray.models.base import BaseModel

logger = logging.getLogger(__name__)


class EmotionAnalysisModel(BaseModel):
    """Emotion analysis using py-feat (Facial Expression Analysis Toolbox)."""

    def __init__(
        self,
        detector: str = "retinaface",
        au_model: str = "xgb",
        emotion_model: str = "resmasknet",
        **kwargs,
    ):
        """Initialize emotion analysis model.

        Args:
            detector: Face detector ('retinaface', 'mtcnn', 'faceboxes')
            au_model: Action Unit model ('svm', 'xgb', 'rf')
            emotion_model: Emotion model ('svm', 'resmasknet')
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.detector_name = detector
        self.au_model_name = au_model
        self.emotion_model_name = emotion_model
        self.detector = None

    def load_model(self) -> None:
        """Load py-feat detector."""
        try:
            from feat import Detector

            logger.info(
                f"Loading py-feat detector: {self.detector_name}, "
                f"AU model: {self.au_model_name}, "
                f"emotion model: {self.emotion_model_name}"
            )

            self.detector = Detector(
                face_model=self.detector_name,
                au_model=self.au_model_name,
                emotion_model=self.emotion_model_name,
            )

            logger.info("py-feat detector loaded successfully")
        except ImportError as ie:
            error_msg = str(ie)
            logger.error(f"Import error: {error_msg}")
            
            # Check for scipy version issue
            if "binom_test" in error_msg or "scipy" in error_msg.lower():
                raise ImportError(
                    "py-feat is not compatible with scipy 1.13+. "
                    "Please install scipy 1.12.0 or earlier:\n"
                    "  pip install 'scipy>=1.10.0,<1.13.0'\n"
                    "Or using uv:\n"
                    "  uv pip install 'scipy>=1.10.0,<1.13.0'\n"
                    f"Original error: {error_msg}"
                )
            else:
                raise ImportError(
                    "py-feat package is required for emotion analysis. "
                    f"Install it with: pip install py-feat\n"
                    f"Original error: {error_msg}"
                )
        except Exception as e:
            logger.error(f"Error loading py-feat detector: {e}")
            raise

    def predict(self, frames: np.ndarray, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run emotion analysis on frame(s).

        Args:
            frames: Input frame(s) as numpy array
                    - Single frame: (H, W, C) in RGB format
                    - Batch of frames: (B, H, W, C) in RGB format
            **kwargs: Additional prediction arguments

        Returns:
            For single frame:
                Dictionary containing emotion analysis results:
                    - faces: List of face detections with emotions
                    - num_faces: Number of faces detected
            For batch:
                List of dictionaries, one per frame in batch
        """
        if self.detector is None:
            self.load_model()

        # Handle both single frame and batched frames
        is_batch = len(frames.shape) == 4
        
        if is_batch:
            # Process batch of frames - return list of results
            batch_results = []
            for frame in frames:
                result = self._detect_single_frame(frame)
                batch_results.append(result)
            return batch_results
        else:
            # Process single frame - return single result dict
            return self._detect_single_frame(frames)
    
    def _detect_single_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect emotions in a single frame.
        
        Args:
            frame: Single frame as numpy array (H, W, C) in RGB format
            
        Returns:
            Dictionary with faces and num_faces
        """
        # Run detection
        results = self.detector.detect_faces(frame)

        # py-feat returns either a DataFrame (faces detected) or empty list (no faces)
        if results is None or len(results) == 0 or isinstance(results, list):
            return {
                "faces": [],
                "num_faces": 0,
            }

        # Parse results (DataFrame)
        faces = []
        for idx in range(len(results)):
            face_data = {
                "face_id": idx,
                "bbox": results.iloc[idx][["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]].tolist()
                if all(
                    col in results.columns
                    for col in ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]
                )
                else None,
                "confidence": results.iloc[idx]["FaceScore"] if "FaceScore" in results.columns else None,
            }

            # Extract emotions if available
            emotion_cols = [col for col in results.columns if col.startswith("emotion_")]
            if emotion_cols:
                emotions = {}
                for col in emotion_cols:
                    emotion_name = col.replace("emotion_", "")
                    emotions[emotion_name] = float(results.iloc[idx][col])
                face_data["emotions"] = emotions

            # Extract Action Units if available
            au_cols = [col for col in results.columns if col.startswith("AU")]
            if au_cols:
                action_units = {}
                for col in au_cols:
                    action_units[col] = float(results.iloc[idx][col])
                face_data["action_units"] = action_units

            faces.append(face_data)

        return {
            "faces": faces,
            "num_faces": len(faces),
        }
