"""
Base OCR utilities for timestamp extraction

Shared logic for parsing timestamps from OCR text
"""

import re
import datetime
from typing import Optional


class TimestampParser:
    """Shared timestamp parsing logic for all OCR implementations"""

    @staticmethod
    def parse_timestamp_from_text(text: str) -> Optional[datetime.datetime]:
        """
        Parse timestamp from OCR-extracted text

        Looks for patterns like:
        - DD.MM.YYYY HH:MM
        - DD.MM.YY HH.MM

        Args:
            text: Text extracted from OCR

        Returns:
            Parsed datetime object, or None if parsing failed
        """
        if not text:
            return None

        # Clean text
        text = text.replace(",", "")

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

        # Failed to parse
        return None
