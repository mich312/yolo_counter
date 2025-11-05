"""
PaddleOCR-based timestamp extraction module

Faster alternative to EasyOCR (3-5x speedup)
Model size: <10MB vs ~100MB for EasyOCR
"""

import datetime
import numpy as np
from typing import Optional, List
from .ocr_base import TimestampParser


class PaddleTimestampExtractor:
    """Extracts timestamp information from images using PaddleOCR (faster)"""

    def __init__(self, languages: List[str]):
        """
        Initialize PaddleOCR reader

        Args:
            languages: List of language codes (e.g., ['de', 'en'])
                      Note: PaddleOCR uses different codes: 'en', 'ch', 'japan', etc.
        """
        self.languages = languages
        self.ocr = None

        # Map common language codes to PaddleOCR format
        self.lang_map = {
            'de': 'german',
            'en': 'en',
            'ch': 'ch',
            'fr': 'french',
            'es': 'spanish',
            'it': 'italian',
        }

    def load_reader(self):
        """Load the PaddleOCR reader (fast initialization)"""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR not installed. Install with: pip install 'paddleocr>=2.8,<3.0'"
            )

        # Use first language (PaddleOCR typically uses single language)
        lang_code = self.languages[0] if self.languages else 'en'
        paddle_lang = self.lang_map.get(lang_code, 'en')

        print(f"Loading PaddleOCR reader with language: {paddle_lang}...")

        # Initialize PaddleOCR with optimized settings
        # Using simple initialization compatible with v2.8.x
        # Note: v2.8.x is more reliable for offline use
        try:
            self.ocr = PaddleOCR(
                lang=paddle_lang,
                use_angle_cls=False,  # Disable angle classification for speed
                use_gpu=False,  # CPU mode (v2.8.x parameter)
                show_log=False,  # Suppress logs (v2.8.x parameter)
            )
        except TypeError:
            # Fallback for v3.x+ API (different parameters)
            print("  Detected PaddleOCR v3.x+ - using simplified initialization")
            print("  Note: First run requires internet to download models")
            self.ocr = PaddleOCR(
                lang=paddle_lang,
                use_angle_cls=False,
            )

        print("✓ PaddleOCR reader loaded successfully!")
        print("  Speed: 3-5x faster than EasyOCR")
        print("  Model size: <10MB")

    def extract_timestamp(self, image: np.ndarray) -> Optional[datetime.datetime]:
        """
        Extract timestamp from image using PaddleOCR

        Args:
            image: Input image (numpy array)

        Returns:
            Parsed datetime object, or None if extraction failed
        """
        if self.ocr is None:
            raise RuntimeError("OCR reader not loaded. Call load_reader() first.")

        try:
            # Run PaddleOCR
            results = self.ocr.ocr(image, cls=False)

            # Extract text from PaddleOCR results
            # Format: [[bbox, (text, confidence)], ...]
            if not results or not results[0]:
                return None

            # Concatenate all recognized text
            text_parts = []
            for line in results[0]:
                if len(line) >= 2:
                    text_parts.append(line[1][0])  # line[1][0] is the text

            text = " ".join(text_parts)

            # Parse timestamp using shared logic
            timestamp = TimestampParser.parse_timestamp_from_text(text)

            if timestamp is None:
                print(f"Failed to parse timestamp from text: {text}")

            return timestamp

        except Exception as e:
            print(f"PaddleOCR extraction error: {e}")
            return None


# Backward compatibility alias
TimestampExtractor = PaddleTimestampExtractor
