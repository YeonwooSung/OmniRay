"""Emotion analysis model using py-feat."""

import logging
from typing import Any, Dict, List, Union
import torch
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


    @torch.no_grad()
    def _detect_single_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect emotions in a single frame.

        Args:
            frame: Single frame as numpy array (H, W, C) in RGB format

        Returns:
            Dictionary with faces and num_faces
        """
        # Ensure frame is proper numpy array with correct dtype
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        
        # Ensure uint8 dtype for image processing
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                # Normalized float image - convert to uint8
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Ensure frame is contiguous in memory
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        try:
            # Step 1: Detect faces - returns list of face bounding boxes
            # Format: [[[x1, y1, x2, y2, score], ...]] for each frame
            face_results = self.detector.detect_faces(frame)
            
            # Check if any faces detected
            if face_results is None or len(face_results) == 0:
                return {"faces": [], "num_faces": 0}
            
            # face_results[0] contains faces for this frame
            frame_faces = face_results[0] if len(face_results) > 0 else []
            if not frame_faces or len(frame_faces) == 0:
                return {"faces": [], "num_faces": 0}
            
            # Step 2: Detect landmarks
            landmarks = self.detector.detect_landmarks(frame, face_results)
            
            # Step 3: Detect Action Units
            aus = self.detector.detect_aus(frame, landmarks)
            
            # Step 4: Detect emotions
            # Returns list of arrays, one per frame, with shape (num_faces, num_emotions)
            emotions = self.detector.detect_emotions(frame, face_results, landmarks)

        except Exception as e:
            logger.warning(f"Detection failed for frame (shape={frame.shape}, dtype={frame.dtype}): {e}")
            return {"faces": [], "num_faces": 0}

        # Parse results
        faces = []
        emotion_names = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
        
        for idx, face_bbox in enumerate(frame_faces):
            face_data = {
                "face_id": idx,
                "bbox": [float(face_bbox[0]), float(face_bbox[1]), 
                        float(face_bbox[2]), float(face_bbox[3])],
                "confidence": float(face_bbox[4]) if len(face_bbox) > 4 else None,
            }
            
            # Extract emotions for this face
            if emotions and len(emotions) > 0 and emotions[0] is not None:
                emotion_array = emotions[0]  # emotions for this frame
                if idx < len(emotion_array):
                    face_emotions = emotion_array[idx]
                    face_data["emotions"] = {
                        name: float(score) 
                        for name, score in zip(emotion_names, face_emotions)
                        if not np.isnan(score)
                    }
            
            # Extract Action Units for this face
            if aus and len(aus) > 0 and aus[0] is not None:
                au_array = aus[0]  # AUs for this frame
                if idx < len(au_array):
                    # AU names typically: AU01, AU02, AU04, etc.
                    au_names = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", 
                               "AU09", "AU10", "AU11", "AU12", "AU14", "AU15",
                               "AU17", "AU20", "AU23", "AU24", "AU25", "AU26",
                               "AU28", "AU43"]
                    face_aus = au_array[idx] if hasattr(au_array, '__getitem__') else au_array
                    if hasattr(face_aus, '__iter__'):
                        action_units = {}
                        for i, val in enumerate(face_aus):
                            if i < len(au_names) and not (isinstance(val, float) and np.isnan(val)):
                                action_units[au_names[i]] = float(val)
                        if action_units:
                            face_data["action_units"] = action_units
            
            faces.append(face_data)

        return {
            "faces": faces,
            "num_faces": len(faces),
        }
