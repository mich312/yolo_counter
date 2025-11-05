"""
OCR (Optical Character Recognition) module

Extracts timestamps from webcam images
"""

import re
import datetime
import numpy as np
from typing import Optional, List
from easyocr import Reader


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
            text = " ".join([result[1] for result in results]).replace(",", "")

            # Extract date pattern: DD.MM.YYYY or DD.MM.YY
            date_match = re.search(r'\d{2}.\d{2}.\d{2,4}', text)
            if not date_match:
                return None

            # Remove date from text to avoid interference with time search
            date_str = date_match.group(0)
            text = text.replace(date_str, "")

            # Extract time pattern: HH:MM or HH.MM
            time_match = re.search(r'\d{2}(:|\.)\d{2}', text)
            if not time_match:
                return None

            time_str = time_match.group(0)

            # Combine date and time
            datetime_str = date_str.replace(" ", ".") + " " + time_str.replace(" ", ".")

            # Try parsing with different formats
            for fmt in ['%d.%m.%Y %H.%M', '%d.%m.%Y %H:%M', '%d.%m.%y %H.%M', '%d.%m.%y %H:%M']:
                try:
                    parsed_date = datetime.datetime.strptime(datetime_str, fmt)
                    return parsed_date
                except ValueError:
                    continue

            print(f"Failed to parse date: {datetime_str}")
            return None

        except Exception as e:
            print(f"OCR extraction error: {e}")
            return None
