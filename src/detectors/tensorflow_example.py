"""
Example: TensorFlow detector implementation

This shows how easy it is to swap detection models!
Just inherit from BaseDetector and implement the required methods.

To use this detector, modify main.py line 50:
    self.detector = TensorFlowDetector(detector_config)
instead of:
    self.detector = YOLODetector(detector_config)
"""

import numpy as np
from typing import List
from .base import BaseDetector, DetectionResult


class TensorFlowDetector(BaseDetector):
    """
    Example TensorFlow detector implementation

    This is a template showing how to integrate a different detection model.
    Replace the stub methods with actual TensorFlow model loading and inference.
    """

    def __init__(self, config: dict):
        """
        Initialize TensorFlow detector

        Args:
            config: Dictionary with model configuration
                - model_path: Path to TensorFlow saved model
                - confidence_threshold: Detection threshold
                - etc.
        """
        super().__init__(config)
        self.model_path = config['model_path']
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        # Add any TensorFlow-specific configuration here

    def load_model(self):
        """
        Load TensorFlow model

        Example implementation:
            import tensorflow as tf
            self.model = tf.saved_model.load(self.model_path)
            self.classes = ['person', 'car', 'dog', ...]  # Your model's classes
        """
        print(f"Loading TensorFlow model from {self.model_path}...")
        # TODO: Implement actual TensorFlow model loading
        # self.model = tf.saved_model.load(self.model_path)
        # self.classes = load_class_names()
        raise NotImplementedError("TensorFlow model loading not implemented yet")

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Perform detection using TensorFlow model

        Example implementation:
            # Preprocess image
            input_tensor = preprocess(image)

            # Run inference
            detections = self.model(input_tensor)

            # Parse results
            results = []
            for det in detections:
                result = DetectionResult(
                    class_id=det['class_id'],
                    class_name=self.classes[det['class_id']],
                    confidence=det['score'],
                    bbox=(det['x'], det['y'], det['w'], det['h'])
                )
                results.append(result)

            return results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # TODO: Implement actual TensorFlow inference
        raise NotImplementedError("TensorFlow detection not implemented yet")

    def get_model_info(self) -> dict:
        """Get TensorFlow model information"""
        return {
            'name': 'TensorFlow Object Detection',
            'framework': 'TensorFlow',
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'num_classes': len(self.classes) if self.classes else 0
        }


# Example: You could also create other detector types!
#
# class PyTorchDetector(BaseDetector):
#     """PyTorch detection model"""
#     pass
#
# class ONNXDetector(BaseDetector):
#     """ONNX Runtime detection model"""
#     pass
#
# class CustomDetector(BaseDetector):
#     """Your custom detection algorithm"""
#     pass
