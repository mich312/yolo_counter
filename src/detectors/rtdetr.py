"""
RT-DETR detector implementation using Ultralytics

RT-DETR (Real-Time Detection Transformer) from Baidu:
- Transformer-based architecture (different from CNN-based YOLO)
- No NMS required (end-to-end detection)
- Stable inference speed regardless of object count
- 53-55% AP on COCO with 108 FPS on T4 GPU
"""

import numpy as np
from typing import List
from .base import BaseDetector, DetectionResult

try:
    from ultralytics import RTDETR
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class RTDETRDetector(BaseDetector):
    """
    RT-DETR object detection implementation using Ultralytics

    RT-DETR is a transformer-based detector that offers:
    - End-to-end detection (no NMS post-processing)
    - Consistent speed regardless of scene complexity
    - Different architecture approach from YOLO (transformers vs CNNs)
    - Good for complex multi-object scenes
    """

    def __init__(self, config: dict):
        """
        Initialize RT-DETR detector

        Args:
            config: Dictionary with keys:
                - model_name: RT-DETR variant (rtdetr-l, rtdetr-x, rtdtrv2-l, rtdtrv2-x)
                - confidence_threshold: Detection confidence threshold (default: 0.5)
                - device: 'cpu' or 'cuda' (default: 'cpu')
        """
        super().__init__(config)

        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics package not installed. "
                "Install with: pip install ultralytics>=8.0.0"
            )

        self.model_name = config.get('model_name', 'rtdetr-l.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.device = config.get('device', 'cpu')

        # Ensure .pt extension
        if not self.model_name.endswith('.pt'):
            self.model_name += '.pt'

    def load_model(self):
        """
        Load RT-DETR model using Ultralytics

        Model will be auto-downloaded if not present
        """
        print(f"Loading RT-DETR model: {self.model_name}...")
        print(f"  Device: {self.device}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Architecture: Transformer-based (DETR)")

        self.model = RTDETR(self.model_name)

        # Get class names from model
        self.classes = list(self.model.names.values())

        print(f"✓ RT-DETR loaded successfully!")
        print(f"  Model: {self.model_name}")
        print(f"  Classes: {len(self.classes)} (COCO dataset)")
        print(f"  Note: No NMS required (end-to-end detection)")

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Perform RT-DETR object detection on image

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
        """Get RT-DETR model information"""
        return {
            'name': 'RT-DETR',
            'framework': 'Ultralytics (Baidu model)',
            'model_file': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'num_classes': len(self.classes),
            'classes': self.classes,
            'architecture': 'Transformer-based (DETR)',
            'description': 'Real-time transformer detector with no NMS required',
            'best_for': 'Complex scenes, consistent performance, multi-object detection',
            'key_features': [
                'End-to-end detection',
                'No NMS post-processing',
                'Stable inference speed',
                'Transformer attention mechanism'
            ],
            'paper': 'DETRs Beat YOLOs on Real-time Object Detection (Baidu, 2023)'
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
def create_rtdetr_detector(variant='l', confidence=0.5, device='cpu'):
    """
    Create an RT-DETR detector with common settings

    Args:
        variant: Model size ('l' or 'x')
        confidence: Confidence threshold (0.0-1.0)
        device: 'cpu' or 'cuda'

    Returns:
        Configured RTDETRDetector instance

    Example:
        >>> detector = create_rtdetr_detector('l', confidence=0.6)
        >>> detector.load_model()
        >>> results = detector.detect(image)
    """
    config = {
        'model_name': f'rtdetr-{variant}.pt',
        'confidence_threshold': confidence,
        'device': device
    }
    return RTDETRDetector(config)
