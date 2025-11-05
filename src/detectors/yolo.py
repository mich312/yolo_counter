"""
YOLO (You Only Look Once) detector implementation

Supports YOLOv3 and YOLOv3-tiny models
"""

import cv2
import numpy as np
from typing import List, Tuple
from .base import BaseDetector, DetectionResult


class YOLODetector(BaseDetector):
    """
    YOLO object detection implementation

    Uses OpenCV's DNN module to run YOLOv3 or YOLOv3-tiny models
    """

    # COCO dataset classes (80 classes)
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    def __init__(self, config: dict):
        """
        Initialize YOLO detector

        Args:
            config: Dictionary with keys:
                - model_path: Path to .weights file
                - config_path: Path to .cfg file
                - confidence_threshold: Detection confidence threshold (default: 0.5)
                - nms_threshold: Non-maximum suppression threshold (default: 0.4)
                - tile_size: Image tile size for segmentation (default: 416)
                - scale_factor: Scale factor for blob (default: 0.00392 * 6)
        """
        super().__init__(config)
        self.model_path = config['model_path']
        self.config_path = config['config_path']
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.4)
        self.tile_size = config.get('tile_size', 416)
        self.scale_factor = config.get('scale_factor', 0.00392 * 6)
        self.classes = self.COCO_CLASSES
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def load_model(self):
        """Load YOLO model using OpenCV DNN"""
        print(f"Loading YOLO model from {self.model_path}...")
        self.model = cv2.dnn.readNet(self.model_path, self.config_path)
        print("YOLO model loaded successfully!")

    def get_model_info(self) -> dict:
        """Get YOLO model information"""
        return {
            'name': 'YOLOv3',
            'framework': 'Darknet (via OpenCV DNN)',
            'model_path': self.model_path,
            'config_path': self.config_path,
            'num_classes': len(self.classes),
            'classes': self.classes,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'tile_size': self.tile_size
        }

    def _get_output_layers(self) -> List[str]:
        """Get output layer names from the network"""
        layer_names = self.model.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
        return output_layers

    def _segment_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], int, int]:
        """
        Segment large image into smaller tiles

        Args:
            image: Input image

        Returns:
            Tuple of (tile_list, tile_width, tile_height)
        """
        count_x = image.shape[1] // self.tile_size
        count_y = image.shape[0] // self.tile_size
        tile_width = image.shape[1] // count_x
        tile_height = image.shape[0] // count_y

        tiles = []
        for x in range(count_x):
            for y in range(count_y):
                tile = image[
                    y*tile_height:min(y*tile_height+tile_height, image.shape[0]),
                    x*tile_width:min(x*tile_width+tile_width, image.shape[1])
                ]
                tiles.append(tile)

        return tiles, tile_width, tile_height

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Perform YOLO object detection on image

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Segment image into tiles for better detection on large images
        tiles, tile_width, tile_height = self._segment_image(image)

        # Prepare blobs for neural network
        blobs = [
            cv2.dnn.blobFromImage(tile, self.scale_factor, (self.tile_size, self.tile_size),
                                 (0, 0, 0), True, crop=False)
            for tile in tiles
        ]

        all_results = []

        # Process each tile
        for tile_idx, blob in enumerate(blobs):
            self.model.setInput(blob)
            outs = self.model.forward(self._get_output_layers())

            # Parse detections
            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.confidence_threshold:
                        center_x = int(detection[0] * tile_width)
                        center_y = int(detection[1] * tile_height)
                        w = int(detection[2] * tile_width)
                        h = int(detection[3] * tile_height)
                        x = center_x - w // 2
                        y = center_y - h // 2

                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            # Apply Non-Maximum Suppression
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                          self.confidence_threshold,
                                          self.nms_threshold)

                for i in indices:
                    box = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]

                    # Create detection result
                    result = DetectionResult(
                        class_id=class_id,
                        class_name=self.classes[class_id] if class_id < len(self.classes) else "unknown",
                        confidence=confidence,
                        bbox=(box[0], box[1], box[2], box[3])
                    )
                    all_results.append(result)

        return all_results

    def draw_detections(self, image: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            image: Input image
            results: List of detection results

        Returns:
            Image with bounding boxes drawn
        """
        output = image.copy()

        for result in results:
            x, y, w, h = result.bbox
            color = self.colors[result.class_id].tolist()

            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f"{result.class_name}: {result.confidence:.2f}"
            cv2.putText(output, label, (x - 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output
