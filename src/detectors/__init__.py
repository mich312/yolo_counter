"""
Detectors module - pluggable detection models

Available detectors:
- YOLODetector: YOLOv3/YOLOv3-tiny via OpenCV DNN
- YOLO11Detector: Latest YOLO11 via Ultralytics (recommended)
- RTDETRDetector: Transformer-based RT-DETR via Ultralytics
"""

from .base import BaseDetector, DetectionResult
from .yolo import YOLODetector

# Import newer detectors if ultralytics is available
try:
    from .yolo11 import YOLO11Detector, create_yolo11_detector
    from .rtdetr import RTDETRDetector, create_rtdetr_detector
    MODERN_DETECTORS_AVAILABLE = True
except ImportError:
    MODERN_DETECTORS_AVAILABLE = False
    YOLO11Detector = None
    RTDETRDetector = None

__all__ = [
    'BaseDetector',
    'DetectionResult',
    'YOLODetector',
    'YOLO11Detector',
    'RTDETRDetector',
    'create_yolo11_detector',
    'create_rtdetr_detector',
    'MODERN_DETECTORS_AVAILABLE'
]
