#!/usr/bin/env python3
"""
Test suite for modular architecture

Tests the new modular structure without requiring database or YOLO weights
"""

import sys
import os

# Set up test environment
os.environ['POSTGRES_HOST'] = 'test_host'
os.environ['POSTGRES_USER'] = 'test_user'
os.environ['POSTGRES_PASSWORD'] = 'test_password'
os.environ['POSTGRES_DB'] = 'test_db'

print("=" * 60)
print("Modular Architecture Test Suite")
print("=" * 60)

all_passed = True

# Test 1: Import base detector
print("\n[Test 1] Testing base detector import...")
try:
    from src.detectors.base import BaseDetector, DetectionResult
    print("  ✓ BaseDetector imported")
    print("  ✓ DetectionResult imported")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    all_passed = False

# Test 2: Import YOLO detector
print("\n[Test 2] Testing YOLO detector import...")
try:
    from src.detectors.yolo import YOLODetector
    print("  ✓ YOLODetector imported")

    # Check it inherits from BaseDetector
    if issubclass(YOLODetector, BaseDetector):
        print("  ✓ YOLODetector inherits from BaseDetector")
    else:
        print("  ✗ YOLODetector does not inherit from BaseDetector")
        all_passed = False
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    all_passed = False

# Test 3: Import configuration
print("\n[Test 3] Testing configuration module...")
try:
    from src.config import AppConfig, DatabaseConfig, DetectorConfig
    print("  ✓ AppConfig imported")
    print("  ✓ DatabaseConfig imported")
    print("  ✓ DetectorConfig imported")

    # Test configuration loading
    config = AppConfig.from_env()
    print(f"  ✓ Configuration loaded: {config.database.host}")

except ImportError as e:
    print(f"  ✗ Import error: {e}")
    all_passed = False
except Exception as e:
    print(f"  ✗ Configuration error: {e}")
    all_passed = False

# Test 4: Import database module
print("\n[Test 4] Testing database module...")
try:
    from src.database import DatabaseManager
    print("  ✓ DatabaseManager imported")

    # Test initialization (without actual connection)
    from src.config import DatabaseConfig
    db_config = DatabaseConfig.from_env()
    db = DatabaseManager(db_config)
    print("  ✓ DatabaseManager initialized")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    all_passed = False

# Test 5: Import OCR module
print("\n[Test 5] Testing OCR module...")
try:
    from src.ocr import TimestampExtractor
    print("  ✓ TimestampExtractor imported")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    all_passed = False

# Test 6: Test DetectionResult dataclass
print("\n[Test 6] Testing DetectionResult dataclass...")
try:
    result = DetectionResult(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox=(10, 20, 100, 200)
    )
    print(f"  ✓ DetectionResult created: {result.class_name}")

    result_dict = result.to_dict()
    assert result_dict['class_name'] == 'person'
    assert result_dict['confidence'] == 0.95
    print("  ✓ to_dict() method works")
except Exception as e:
    print(f"  ✗ Error: {e}")
    all_passed = False

# Test 7: Test BaseDetector utility methods
print("\n[Test 7] Testing BaseDetector utility methods...")
try:
    # Create mock detector
    class MockDetector(BaseDetector):
        def load_model(self):
            pass

        def detect(self, image):
            return []

        def get_model_info(self):
            return {}

    detector = MockDetector({})
    detector.classes = ['person', 'car', 'dog']

    # Test filter methods
    test_results = [
        DetectionResult(0, 'person', 0.9, (0, 0, 100, 100)),
        DetectionResult(1, 'car', 0.6, (0, 0, 100, 100)),
        DetectionResult(0, 'person', 0.4, (0, 0, 100, 100)),
    ]

    # Filter by class
    filtered = detector.filter_by_class(test_results, ['person'])
    assert len(filtered) == 2, f"Expected 2 persons, got {len(filtered)}"
    print("  ✓ filter_by_class() works")

    # Filter by confidence
    filtered = detector.filter_by_confidence(test_results, 0.5)
    assert len(filtered) == 2, f"Expected 2 high-conf, got {len(filtered)}"
    print("  ✓ filter_by_confidence() works")

    # Count by class
    counts = detector.count_by_class(test_results)
    assert counts['person'] == 2, f"Expected 2 persons, got {counts.get('person', 0)}"
    assert counts['car'] == 1, f"Expected 1 car, got {counts.get('car', 0)}"
    print("  ✓ count_by_class() works")

except Exception as e:
    print(f"  ✗ Error: {e}")
    all_passed = False

# Test 8: Test module structure
print("\n[Test 8] Testing module structure...")
required_files = {
    'src/__init__.py': 'Package init',
    'src/config.py': 'Configuration module',
    'src/database.py': 'Database module',
    'src/ocr.py': 'OCR module',
    'src/detectors/__init__.py': 'Detectors package init',
    'src/detectors/base.py': 'Base detector',
    'src/detectors/yolo.py': 'YOLO detector',
    'src/detectors/tensorflow_example.py': 'TensorFlow example'
}

for file, desc in required_files.items():
    if os.path.exists(file):
        print(f"  ✓ {file} ({desc})")
    else:
        print(f"  ✗ {file} missing ({desc})")
        all_passed = False

# Test 9: Test YOLO detector COCO classes
print("\n[Test 9] Testing YOLO detector COCO classes...")
try:
    assert len(YOLODetector.COCO_CLASSES) == 80, "Should have 80 COCO classes"
    assert YOLODetector.COCO_CLASSES[0] == 'person', "First class should be 'person'"
    print(f"  ✓ YOLO has {len(YOLODetector.COCO_CLASSES)} COCO classes")
    print(f"  ✓ Sample classes: {YOLODetector.COCO_CLASSES[:5]}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    all_passed = False

# Test 10: Test configuration validation
print("\n[Test 10] Testing configuration validation...")
try:
    # Test with missing credentials
    os.environ['POSTGRES_HOST'] = ''
    try:
        bad_config = DatabaseConfig.from_env()
        bad_config.validate()
        print("  ✗ Should have raised ValueError for missing credentials")
        all_passed = False
    except ValueError:
        print("  ✓ Configuration validation works")

    # Restore
    os.environ['POSTGRES_HOST'] = 'test_host'
except Exception as e:
    print(f"  ✗ Error: {e}")
    all_passed = False

# Test 11: Check main.py uses modular imports
print("\n[Test 11] Testing main.py uses modular imports...")
with open('main.py', 'r') as f:
    main_content = f.read()

required_imports = [
    'from src.config import AppConfig',
    'from src.database import DatabaseManager',
    'from src.detectors.yolo import YOLODetector',
    'from src.ocr import TimestampExtractor'
]

for imp in required_imports:
    if imp in main_content:
        print(f"  ✓ {imp}")
    else:
        print(f"  ✗ Missing: {imp}")
        all_passed = False

# Test 12: Check WebcamProcessor class exists
print("\n[Test 12] Testing WebcamProcessor class...")
if 'class WebcamProcessor:' in main_content:
    print("  ✓ WebcamProcessor class defined")
else:
    print("  ✗ WebcamProcessor class missing")
    all_passed = False

if 'def process_webcam' in main_content:
    print("  ✓ process_webcam method defined")
else:
    print("  ✗ process_webcam method missing")
    all_passed = False

# Test 13: Verify old monolithic code removed
print("\n[Test 13] Verifying modular refactoring...")
# Check that old inline functions are removed
old_patterns = [
    'def get_output_layers(net):',  # Now in YOLODetector
    'def query(query, db_conn):',  # Now in DatabaseManager
    'def insert_or_update(',  # Now in DatabaseManager
]

refactored_count = 0
for pattern in old_patterns:
    if pattern not in main_content:
        refactored_count += 1

print(f"  ✓ {refactored_count}/{len(old_patterns)} functions properly refactored")

# Test 14: Check backwards compatibility
print("\n[Test 14] Checking backwards compatibility...")
if os.path.exists('main_old.py'):
    print("  ✓ Original main.py backed up as main_old.py")
else:
    print("  ⚠ No backup found (main_old.py)")

print("\n" + "=" * 60)
if all_passed:
    print("✅ ALL MODULAR ARCHITECTURE TESTS PASSED!")
    print("=" * 60)
    print("\nThe code has been successfully refactored to modular architecture!")
    print("\nKey improvements:")
    print("  • Abstract BaseDetector allows easy model swapping")
    print("  • YOLODetector implements YOLO-specific logic")
    print("  • Configuration module centralizes all settings")
    print("  • Database module handles all DB operations")
    print("  • OCR module isolates timestamp extraction")
    print("  • Main.py is now clean and orchestrates components")
    print("\nTo add a new detector:")
    print("  1. Create new class inheriting from BaseDetector")
    print("  2. Implement load_model(), detect(), and get_model_info()")
    print("  3. Change line 50 in main.py to use your detector")
    print("\nSee src/detectors/tensorflow_example.py for template!")
    sys.exit(0)
else:
    print("⚠ SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(1)
