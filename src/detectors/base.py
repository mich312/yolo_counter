"""
Base detector class - abstract interface for all detection models

This allows easy swapping of different detection models (YOLO, TensorFlow, PyTorch, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2


@dataclass
class DetectionResult:
    """Result from object detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox
        }


class BaseDetector(ABC):
    """
    Abstract base class for all object detection models

    Any detection model (YOLO, TensorFlow, PyTorch, etc.) should inherit from this
    and implement the required methods.
    """

    def __init__(self, config: dict):
        """
        Initialize the detector

        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.model = None
        self.classes = []

    @abstractmethod
    def load_model(self):
        """
        Load the detection model

        This should load weights, initialize the model, and prepare it for inference.
        Called once during initialization for performance.
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Perform object detection on an image

        Args:
            image: Input image as numpy array (BGR format from OpenCV)

        Returns:
            List of DetectionResult objects containing detected objects
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model metadata (name, version, classes, etc.)
        """
        pass

    def filter_by_class(self, results: List[DetectionResult],
                       class_names: List[str]) -> List[DetectionResult]:
        """
        Filter detection results by class names

        Args:
            results: List of detection results
            class_names: List of class names to keep

        Returns:
            Filtered list of detection results
        """
        return [r for r in results if r.class_name in class_names]

    def filter_by_confidence(self, results: List[DetectionResult],
                           threshold: float) -> List[DetectionResult]:
        """
        Filter detection results by confidence threshold

        Args:
            results: List of detection results
            threshold: Minimum confidence threshold (0.0 to 1.0)

        Returns:
            Filtered list of detection results
        """
        return [r for r in results if r.confidence >= threshold]

    def count_by_class(self, results: List[DetectionResult]) -> dict:
        """
        Count detections by class name

        Args:
            results: List of detection results

        Returns:
            Dictionary mapping class names to counts
        """
        counts = {}
        for result in results:
            counts[result.class_name] = counts.get(result.class_name, 0) + 1
        return counts

    def draw_detections(self, image: np.ndarray, results: List[DetectionResult],
                       colors: np.ndarray = None) -> np.ndarray:
        """
        Draw bounding boxes on image (shared implementation)

        Args:
            image: Input image
            results: List of detection results
            colors: Optional color array for classes (BGR format)

        Returns:
            Image with bounding boxes drawn
        """
        output = image.copy()

        # Generate default colors if not provided
        if colors is None:
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)

        for result in results:
            x, y, w, h = result.bbox

            # Use color for this class (or default to green)
            if result.class_id < len(colors):
                color = colors[result.class_id].tolist()
            else:
                color = (0, 255, 0)  # Default green

            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f"{result.class_name}: {result.confidence:.2f}"
            cv2.putText(output, label, (x - 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output
