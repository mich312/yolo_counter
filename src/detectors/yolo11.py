"""
YOLO11 detector implementation using Ultralytics

YOLO11 is the latest model from Ultralytics (October 2024):
- 22% fewer parameters than YOLOv8
- Higher accuracy across all model sizes
- Optimized for both CPU and GPU
- Best for detecting people/skiers at varying distances
"""

import numpy as np
from typing import List
from .base import BaseDetector, DetectionResult

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class YOLO11Detector(BaseDetector):
    """
    YOLO11 object detection implementation using Ultralytics

    Uses the latest YOLO11 models which offer:
    - Superior accuracy with fewer parameters
    - Optimized inference speed (13.5ms)
    - Better small object detection (distant skiers)
    - Easy integration via Ultralytics library
    """

    def __init__(self, config: dict):
        """
        Initialize YOLO11 detector

        Args:
            config: Dictionary with keys:
                - model_name: YOLO11 variant (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
                - confidence_threshold: Detection confidence threshold (default: 0.5)
                - device: 'cpu' or 'cuda' (default: 'cpu')
        """
        super().__init__(config)

        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics package not installed. "
                "Install with: pip install ultralytics>=8.3.0"
            )

        self.model_name = config.get('model_name', 'yolo11n.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.device = config.get('device', 'cpu')

        # Ensure .pt extension
        if not self.model_name.endswith('.pt'):
            self.model_name += '.pt'

    def load_model(self):
        """
        Load YOLO11 model using Ultralytics

        Model will be auto-downloaded if not present
        """
        print(f"Loading YOLO11 model: {self.model_name}...")
        print(f"  Device: {self.device}")
        print(f"  Confidence threshold: {self.confidence_threshold}")

        self.model = YOLO(self.model_name)

        # Get class names from model
        self.classes = list(self.model.names.values())

        print(f"✓ YOLO11 loaded successfully!")
        print(f"  Model: {self.model_name}")
        print(f"  Classes: {len(self.classes)} (COCO dataset)")

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Perform YOLO11 object detection on image

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False  # Suppress output
        )

        # Parse results
        detections = []

        for result in results:
            boxes = result.boxes

            for i in range(len(boxes)):
                # Extract box data
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Convert to [x, y, w, h] format
                x1, y1, x2, y2 = box
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                # Get class name
                class_name = self.classes[cls_id] if cls_id < len(self.classes) else "unknown"

                # Create detection result
                detection = DetectionResult(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x, y, w, h)
                )
                detections.append(detection)

        return detections

    def get_model_info(self) -> dict:
        """Get YOLO11 model information"""
        return {
            'name': 'YOLO11',
            'framework': 'Ultralytics',
            'model_file': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'num_classes': len(self.classes),
            'classes': self.classes,
            'description': 'Latest YOLO11 model (Oct 2024) with 22% fewer params and higher accuracy',
            'best_for': 'Real-time detection, person/skier tracking, small object detection',
            'release_date': 'October 2024'
        }

    def draw_detections(self, image: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            image: Input image
            results: List of detection results

        Returns:
            Image with bounding boxes drawn
        """
        import cv2

        output = image.copy()

        # Generate colors for each class
        np.random.seed(42)  # Consistent colors
        colors = {}
        for result in results:
            if result.class_id not in colors:
                colors[result.class_id] = tuple(map(int, np.random.randint(0, 255, 3)))

        for result in results:
            x, y, w, h = result.bbox
            color = colors[result.class_id]

            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Draw label with background
            label = f"{result.class_name}: {result.confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Label background
            cv2.rectangle(output, (x, y - label_h - 10), (x + label_w, y), color, -1)

            # Label text
            cv2.putText(
                output, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        return output


# Convenience function for quick model creation
def create_yolo11_detector(variant='n', confidence=0.5, device='cpu'):
    """
    Create a YOLO11 detector with common settings

    Args:
        variant: Model size ('n', 's', 'm', 'l', 'x')
        confidence: Confidence threshold (0.0-1.0)
        device: 'cpu' or 'cuda'

    Returns:
        Configured YOLO11Detector instance

    Example:
        >>> detector = create_yolo11_detector('n', confidence=0.6)
        >>> detector.load_model()
        >>> results = detector.detect(image)
    """
    config = {
        'model_name': f'yolo11{variant}.pt',
        'confidence_threshold': confidence,
        'device': device
    }
    return YOLO11Detector(config)
