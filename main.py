#!/usr/bin/env python3
"""
YOLO Counter - Modular Object Detection System

Detects objects in webcam images and stores results in database.
Now with modular architecture for easy model swapping!
"""

import cv2
import numpy as np
import urllib.request
import datetime
import time
import os
import pandas as pd
from typing import Optional

# Import our modular components
from src.config import AppConfig
from src.database import DatabaseManager
from src.detectors.yolo import YOLODetector
from src.ocr import TimestampExtractor as EasyOCRExtractor
from src.ocr_paddle import PaddleTimestampExtractor


class WebcamProcessor:
    """Main processor for webcam detection"""

    def __init__(self, config: AppConfig):
        """
        Initialize processor with configuration

        Args:
            config: Application configuration
        """
        self.config = config

        # Initialize components
        print("Initializing components...")
        self.db = DatabaseManager(config.database)

        # Initialize detector (YOLO by default, but can be swapped!)
        detector_config = {
            'model_path': config.detector.model_path,
            'config_path': config.detector.config_path,
            'confidence_threshold': config.detector.confidence_threshold,
            'nms_threshold': config.detector.nms_threshold,
            'tile_size': config.detector.tile_size,
            'scale_factor': config.detector.scale_factor
        }
        self.detector = YOLODetector(detector_config)
        self.detector.load_model()

        # Initialize OCR (if enabled)
        self.ocr = None
        if config.ocr.enabled:
            # Select OCR engine based on config
            if config.ocr.engine == 'paddle':
                print("Using PaddleOCR (3-5x faster)")
                self.ocr = PaddleTimestampExtractor(config.ocr.languages)
            else:
                print("Using EasyOCR (legacy)")
                self.ocr = EasyOCRExtractor(config.ocr.languages)
            self.ocr.load_reader()

        print("Initialization complete!")
        print(f"Detector: {self.detector.get_model_info()['name']}")
        print(f"Detection classes: {config.detector.detection_classes}")

    def fetch_image(self, url: str) -> Optional[np.ndarray]:
        """
        Fetch image from URL

        Args:
            url: Image URL

        Returns:
            Image as numpy array, or None if fetch failed
        """
        try:
            resp = urllib.request.urlopen(url)
            image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"Failed to fetch image from {url}: {e}")
            return None

    def extract_timestamp(self, image: np.ndarray) -> Optional[datetime.datetime]:
        """
        Extract timestamp from image using OCR

        Args:
            image: Input image

        Returns:
            Extracted datetime, or None if extraction failed
        """
        if self.ocr is None:
            return None

        return self.ocr.extract_timestamp(image)

    def save_images(self, webcam_id: str, image: np.ndarray,
                   annotated_image: np.ndarray, timestamp: datetime.datetime):
        """
        Save raw and annotated images to disk

        Args:
            webcam_id: Webcam UUID
            image: Raw image
            annotated_image: Image with detection boxes
            timestamp: Image timestamp
        """
        folder = f"{self.config.storage.image_folder}/{webcam_id}"
        os.makedirs(folder, exist_ok=True)

        # Save raw image
        raw_path = f"{folder}/raw_{timestamp}.jpg"
        cv2.imwrite(raw_path, image)

        # Save annotated image
        annotated_path = f"{folder}/{timestamp}.jpg"
        cv2.imwrite(annotated_path, annotated_image)

    def process_webcam(self, webcam_id: str, webcam_url: str) -> Optional[dict]:
        """
        Process a single webcam

        Args:
            webcam_id: Webcam UUID
            webcam_url: Webcam image URL

        Returns:
            Detection result dictionary, or None if processing failed
        """
        start_time = time.time()

        try:
            # Fetch image
            image = self.fetch_image(webcam_url)
            if image is None:
                return None

            # Extract timestamp
            timestamp = self.extract_timestamp(image)
            if timestamp is None:
                timestamp = datetime.datetime.now()
                print(f"⚠ Could not extract timestamp, using current time: {timestamp}")

            # Perform detection
            detections = self.detector.detect(image)

            # Filter by configured classes
            filtered_detections = self.detector.filter_by_class(
                detections,
                self.config.detector.detection_classes
            )

            # Count detections
            detection_count = len(filtered_detections)
            counts_by_class = self.detector.count_by_class(filtered_detections)

            # Draw boxes on image
            annotated_image = self.detector.draw_detections(image, filtered_detections)

            # Save images
            self.save_images(webcam_id, image, annotated_image, timestamp)

            # Calculate processing time
            elapsed = time.time() - start_time

            # Print results
            print(f"✓ {webcam_url}")
            print(f"  Detections: {detection_count} {counts_by_class}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Processing time: {elapsed:.2f}s")

            return {
                'webcam': webcam_id,
                'date': timestamp,
                'detections': detection_count
            }

        except Exception as e:
            print(f"✗ Error processing {webcam_url}: {e}")
            return None

    def run(self):
        """Main processing loop"""
        print("\n" + "=" * 60)
        print("Starting webcam processing...")
        print("=" * 60 + "\n")

        # Get active webcams from database
        webcams = self.db.get_active_webcams()
        print(f"Found {len(webcams)} active webcam(s)\n")

        # Process each webcam
        results = []
        for idx, row in webcams.iterrows():
            webcam_id = row['id']
            webcam_url = row['url']
            webcam_name = row.get('name', 'Unknown')

            print(f"[{idx + 1}/{len(webcams)}] Processing: {webcam_name}")

            result = self.process_webcam(webcam_id, webcam_url)
            if result:
                results.append(result)

            print()

        # Store results in database
        if results:
            print(f"Storing {len(results)} result(s) in database...")
            df_results = pd.DataFrame(results)
            self.db.store_detections(df_results)
            print("✓ Results stored successfully!")
            print("\nResults summary:")
            print(df_results)
        else:
            print("⚠ No results to store")

        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)


def main():
    """Main entry point"""
    try:
        # Load configuration
        print("Loading configuration...")
        config = AppConfig.from_env()
        print("✓ Configuration loaded\n")

        # Create processor and run
        processor = WebcamProcessor(config)
        processor.run()

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease check your .env file and ensure all required variables are set.")
        print("See .env.example for reference.")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
