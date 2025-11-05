"""
OCR (Optical Character Recognition) module

Extracts timestamps from webcam images
"""

import datetime
import numpy as np
from typing import Optional, List
from easyocr import Reader
from .ocr_base import TimestampParser


class TimestampExtractor:
    """Extracts timestamp information from images using OCR"""

    def __init__(self, languages: List[str]):
        """
        Initialize OCR reader

        Args:
            languages: List of language codes (e.g., ['de', 'en'])
        """
        self.languages = languages
        self.reader = None

    def load_reader(self):
        """Load the OCR reader (can be slow, call once at startup)"""
        print(f"Loading OCR reader with languages: {self.languages}...")
        self.reader = Reader(self.languages)
        print("OCR reader loaded successfully!")

    def extract_timestamp(self, image: np.ndarray) -> Optional[datetime.datetime]:
        """
        Extract timestamp from image

        Args:
            image: Input image (numpy array)

        Returns:
            Parsed datetime object, or None if extraction failed
        """
        if self.reader is None:
            raise RuntimeError("OCR reader not loaded. Call load_reader() first.")

        try:
            # Run OCR
            results = self.reader.readtext(image)

            # Concatenate all text
            text = " ".join([result[1] for result in results])

            # Parse timestamp using shared logic
            timestamp = TimestampParser.parse_timestamp_from_text(text)

            if timestamp is None:
                print(f"Failed to parse timestamp from text: {text}")

            return timestamp

        except Exception as e:
            print(f"OCR extraction error: {e}")
            return None
