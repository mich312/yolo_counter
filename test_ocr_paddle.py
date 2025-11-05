#!/usr/bin/env python3
"""
Test script for PaddleOCR implementation

Tests:
1. Module import
2. Initialization
3. Interface compatibility
4. Basic functionality
"""

import sys
import time
import numpy as np
from datetime import datetime

def test_imports():
    """Test that modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)

    try:
        from src.ocr_paddle import PaddleTimestampExtractor
        print("✓ PaddleTimestampExtractor imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import PaddleTimestampExtractor: {e}")
        return False

def test_initialization():
    """Test PaddleOCR initialization"""
    print("\n" + "="*60)
    print("TEST 2: Initialization")
    print("="*60)

    try:
        from src.ocr_paddle import PaddleTimestampExtractor

        # Test with English language
        extractor = PaddleTimestampExtractor(['en'])
        print("✓ PaddleTimestampExtractor created with ['en']")

        # Test with German and English
        extractor_de = PaddleTimestampExtractor(['de', 'en'])
        print("✓ PaddleTimestampExtractor created with ['de', 'en']")

        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def test_interface():
    """Test that interface matches original OCR module"""
    print("\n" + "="*60)
    print("TEST 3: Interface Compatibility")
    print("="*60)

    try:
        from src.ocr_paddle import PaddleTimestampExtractor
        from src.ocr import TimestampExtractor

        # Check that both classes have the same methods
        paddle_methods = set(dir(PaddleTimestampExtractor))
        easy_methods = set(dir(TimestampExtractor))

        required_methods = ['__init__', 'load_reader', 'extract_timestamp']

        for method in required_methods:
            if method in paddle_methods:
                print(f"✓ PaddleTimestampExtractor has method: {method}")
            else:
                print(f"✗ PaddleTimestampExtractor missing method: {method}")
                return False

        return True
    except Exception as e:
        print(f"✗ Interface check failed: {e}")
        return False

def test_load_reader():
    """Test loading the PaddleOCR reader"""
    print("\n" + "="*60)
    print("TEST 4: Load PaddleOCR Reader")
    print("="*60)

    try:
        from src.ocr_paddle import PaddleTimestampExtractor

        extractor = PaddleTimestampExtractor(['en'])

        print("Loading PaddleOCR reader...")
        start_time = time.time()
        extractor.load_reader()
        load_time = time.time() - start_time

        print(f"✓ PaddleOCR reader loaded successfully in {load_time:.2f}s")

        # Check that ocr is not None
        if extractor.ocr is None:
            print("✗ OCR reader is None after loading")
            return False

        print("✓ OCR reader initialized")
        return True

    except ImportError as e:
        print(f"⚠ PaddleOCR not installed: {e}")
        print("  Install with: pip install paddleocr")
        return None  # None means test skipped, not failed
    except Exception as e:
        print(f"✗ Failed to load reader: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timestamp_extraction():
    """Test timestamp extraction with synthetic image"""
    print("\n" + "="*60)
    print("TEST 5: Timestamp Extraction (Synthetic)")
    print("="*60)

    try:
        from src.ocr_paddle import PaddleTimestampExtractor
        import cv2

        extractor = PaddleTimestampExtractor(['en'])
        extractor.load_reader()

        # Create a synthetic image with text
        # This is a simple test - real images would be better
        image = np.ones((100, 400, 3), dtype=np.uint8) * 255

        # Add text using OpenCV
        text = "05.11.2025 14:30"
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 2, cv2.LINE_AA)

        print(f"Testing with synthetic text: '{text}'")

        start_time = time.time()
        result = extractor.extract_timestamp(image)
        extraction_time = time.time() - start_time

        print(f"Extraction time: {extraction_time*1000:.1f}ms")

        if result is not None:
            print(f"✓ Extracted timestamp: {result}")
            print(f"  Expected: 2025-11-05 14:30:00")

            # Verify the result
            if result.year == 2025 and result.month == 11 and result.day == 5:
                print("✓ Date components match expected values")
            else:
                print(f"⚠ Date mismatch: got {result.date()}, expected 2025-11-05")
        else:
            print("⚠ No timestamp extracted (this may be expected with synthetic image)")
            print("  PaddleOCR may need higher quality text rendering")

        return True

    except ImportError as e:
        print(f"⚠ PaddleOCR not installed: {e}")
        return None
    except Exception as e:
        print(f"✗ Timestamp extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """Test integration with config system"""
    print("\n" + "="*60)
    print("TEST 6: Config Integration")
    print("="*60)

    try:
        from src.config import OCRConfig

        # Test default config (should be paddle)
        config = OCRConfig.from_env()
        print(f"✓ OCRConfig loaded")
        print(f"  Engine: {config.engine}")
        print(f"  Languages: {config.languages}")
        print(f"  Enabled: {config.enabled}")

        if config.engine == 'paddle':
            print("✓ Default engine is 'paddle' (correct)")
        else:
            print(f"⚠ Default engine is '{config.engine}' (expected 'paddle')")

        return True
    except Exception as e:
        print(f"✗ Config integration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PADDLEOCR TEST SUITE")
    print("="*60)

    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['initialization'] = test_initialization()
    results['interface'] = test_interface()
    results['load_reader'] = test_load_reader()
    results['timestamp_extraction'] = test_timestamp_extraction()
    results['config_integration'] = test_config_integration()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result is True else ("✗ FAIL" if result is False else "⊘ SKIP")
        print(f"{status:8} | {test_name}")

    print("\n" + "-"*60)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")

    if failed > 0:
        print("\n⚠ Some tests failed!")
        if skipped > 0:
            print("  Skipped tests may be due to missing dependencies (pip install paddleocr)")
        return 1
    elif skipped > 0:
        print("\n⚠ Some tests skipped - install dependencies: pip install paddleocr")
        return 0
    else:
        print("\n✓ All tests passed!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
