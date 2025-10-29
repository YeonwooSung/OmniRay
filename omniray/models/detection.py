"""Object detection model using YOLOv8."""

import logging
from typing import Any, Dict

import numpy as np

from omniray.models.base import BaseModel

logger = logging.getLogger(__name__)


class ObjectDetectionModel(BaseModel):
    """Object detection using YOLOv8 (Ultralytics)."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs,
    ):
        """Initialize object detection model.

        Args:
            model_name: YOLOv8 model name (e.g., 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None

    def load_model(self) -> None:
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLOv8 model: {self.model_name}")
            self.model = YOLO(self.model_name)
            logger.info("YOLOv8 model loaded successfully")
        except ImportError:
            raise ImportError(
                "ultralytics package is required for object detection. "
                "Install it with: pip install ultralytics"
            )

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Run object detection on frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in RGB format
            **kwargs: Additional prediction arguments

        Returns:
            Dictionary containing detection results:
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - scores: Confidence scores
                - class_ids: Class IDs
                - class_names: Class names
        """
        if self.model is None:
            self.load_model()

        # Run inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            **kwargs,
        )

        # Extract results from first image (batch size 1)
        result = results[0]

        # Parse detections
        boxes = []
        scores = []
        class_ids = []
        class_names = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy().tolist()
            class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
            class_names = [result.names[cls_id] for cls_id in class_ids]

        return {
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "class_names": class_names,
            "num_detections": len(boxes),
        }
