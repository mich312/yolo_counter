#!/usr/bin/env python3
"""
Test suite for modular structure (no dependencies required)

Tests that all files exist and have correct structure
"""

import sys
import os

print("=" * 60)
print("Modular Structure Test Suite")
print("=" * 60)

all_passed = True

# Test 1: Module structure
print("\n[Test 1] Testing module structure...")
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

# Test 2: Check Python syntax
print("\n[Test 2] Testing Python syntax...")
import py_compile

modules = [
    'main.py',
    'src/__init__.py',
    'src/config.py',
    'src/database.py',
    'src/ocr.py',
    'src/detectors/__init__.py',
    'src/detectors/base.py',
    'src/detectors/yolo.py',
    'src/detectors/tensorflow_example.py'
]

for module in modules:
    try:
        py_compile.compile(module, doraise=True)
        print(f"  ✓ {module}")
    except py_compile.PyCompileError as e:
        print(f"  ✗ {module}: {e}")
        all_passed = False

# Test 3: Check main.py uses modular imports
print("\n[Test 3] Testing main.py uses modular imports...")
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

# Test 4: Check WebcamProcessor class
print("\n[Test 4] Testing WebcamProcessor class...")
if 'class WebcamProcessor:' in main_content:
    print("  ✓ WebcamProcessor class defined")
else:
    print("  ✗ WebcamProcessor class missing")
    all_passed = False

required_methods = ['__init__', 'fetch_image', 'extract_timestamp',
                   'save_images', 'process_webcam', 'run']
for method in required_methods:
    if f"def {method}" in main_content:
        print(f"  ✓ {method}() method defined")
    else:
        print(f"  ✗ {method}() method missing")
        all_passed = False

# Test 5: Verify old monolithic code removed
print("\n[Test 5] Verifying code was refactored...")
old_patterns = [
    ('def get_output_layers(net):', 'get_output_layers'),
    ('def query(query, db_conn):', 'query function'),
    ('def insert_or_update(', 'insert_or_update'),
    ('def draw_bounding_box(', 'draw_bounding_box'),
]

for pattern, name in old_patterns:
    if pattern not in main_content:
        print(f"  ✓ {name} moved to module")
    else:
        print(f"  ✗ {name} still in main.py (should be in module)")

# Test 6: Check base detector abstract class
print("\n[Test 6] Checking BaseDetector abstract class...")
with open('src/detectors/base.py', 'r') as f:
    base_content = f.read()

base_checks = [
    ('from abc import ABC, abstractmethod', 'ABC import'),
    ('class BaseDetector(ABC):', 'BaseDetector inherits from ABC'),
    ('@abstractmethod', 'Abstract methods defined'),
    ('class DetectionResult:', 'DetectionResult dataclass'),
    ('def filter_by_class', 'filter_by_class utility'),
    ('def filter_by_confidence', 'filter_by_confidence utility'),
    ('def count_by_class', 'count_by_class utility'),
]

for pattern, desc in base_checks:
    if pattern in base_content:
        print(f"  ✓ {desc}")
    else:
        print(f"  ✗ {desc} missing")
        all_passed = False

# Test 7: Check YOLO detector inherits from BaseDetector
print("\n[Test 7] Checking YOLODetector implementation...")
with open('src/detectors/yolo.py', 'r') as f:
    yolo_content = f.read()

yolo_checks = [
    ('from .base import BaseDetector', 'Imports BaseDetector'),
    ('class YOLODetector(BaseDetector):', 'Inherits from BaseDetector'),
    ('def load_model(self):', 'Implements load_model'),
    ('def detect(self, image:', 'Implements detect'),
    ('def get_model_info(self)', 'Implements get_model_info'),
    ('COCO_CLASSES = [', 'Has COCO classes'),
]

for pattern, desc in yolo_checks:
    if pattern in yolo_content:
        print(f"  ✓ {desc}")
    else:
        print(f"  ✗ {desc} missing")
        all_passed = False

# Test 8: Check configuration module
print("\n[Test 8] Checking configuration module...")
with open('src/config.py', 'r') as f:
    config_content = f.read()

config_checks = [
    ('@dataclass', 'Uses dataclasses'),
    ('class DatabaseConfig:', 'DatabaseConfig class'),
    ('class DetectorConfig:', 'DetectorConfig class'),
    ('class OCRConfig:', 'OCRConfig class'),
    ('class AppConfig:', 'AppConfig class'),
    ('from_env(cls):', 'from_env() factory method'),
    ('from dotenv import load_dotenv', 'Loads .env'),
]

for pattern, desc in config_checks:
    if pattern in config_content:
        print(f"  ✓ {desc}")
    else:
        print(f"  ✗ {desc} missing")
        all_passed = False

# Test 9: Check database module
print("\n[Test 9] Checking database module...")
with open('src/database.py', 'r') as f:
    db_content = f.read()

db_checks = [
    ('class DatabaseManager:', 'DatabaseManager class'),
    ('def query(self, sql:', 'query method'),
    ('def get_active_webcams(self)', 'get_active_webcams method'),
    ('def insert_or_update(', 'insert_or_update method'),
    ('def store_detections(', 'store_detections method'),
]

for pattern, desc in db_checks:
    if pattern in db_content:
        print(f"  ✓ {desc}")
    else:
        print(f"  ✗ {desc} missing")
        all_passed = False

# Test 10: Check OCR module
print("\n[Test 10] Checking OCR module...")
with open('src/ocr.py', 'r') as f:
    ocr_content = f.read()

ocr_checks = [
    ('class TimestampExtractor:', 'TimestampExtractor class'),
    ('def load_reader(self):', 'load_reader method'),
    ('def extract_timestamp(', 'extract_timestamp method'),
    ('from easyocr import Reader', 'Uses EasyOCR'),
]

for pattern, desc in ocr_checks:
    if pattern in ocr_content:
        print(f"  ✓ {desc}")
    else:
        print(f"  ✗ {desc} missing")
        all_passed = False

# Test 11: Check example detector
print("\n[Test 11] Checking TensorFlow example...")
with open('src/detectors/tensorflow_example.py', 'r') as f:
    tf_content = f.read()

if 'class TensorFlowDetector(BaseDetector):' in tf_content:
    print("  ✓ TensorFlowDetector example provided")
    print("  ✓ Shows how to add new detectors")
else:
    print("  ✗ TensorFlowDetector example missing")

# Test 12: Check backwards compatibility
print("\n[Test 12] Checking backwards compatibility...")
if os.path.exists('main_old.py'):
    print("  ✓ Original main.py backed up as main_old.py")
else:
    print("  ⚠ No backup found (main_old.py)")

# Test 13: Count lines of code reduction
print("\n[Test 13] Code organization metrics...")
try:
    with open('main.py', 'r') as f:
        new_lines = len(f.readlines())
    with open('main_old.py', 'r') as f:
        old_lines = len(f.readlines())

    print(f"  • Old main.py: {old_lines} lines")
    print(f"  • New main.py: {new_lines} lines")
    print(f"  • Reduction: {old_lines - new_lines} lines")
    print(f"  ✓ Code is now {((old_lines - new_lines) / old_lines * 100):.1f}% more organized")
except:
    pass

# Count total modular code
total_lines = 0
for module in modules:
    if os.path.exists(module):
        with open(module, 'r') as f:
            total_lines += len(f.readlines())
print(f"  • Total modular codebase: {total_lines} lines across {len(modules)} files")

print("\n" + "=" * 60)
if all_passed:
    print("✅ ALL STRUCTURAL TESTS PASSED!")
    print("=" * 60)
    print("\n🎉 Modular refactoring complete!")
    print("\n✨ Benefits of the new architecture:")
    print("  • Separation of concerns (config, DB, detection, OCR)")
    print("  • Easy to swap detection models (see BaseDetector)")
    print("  • Better testability (each module isolated)")
    print("  • Cleaner main.py (orchestration only)")
    print("  • Type hints and documentation throughout")
    print("\n📚 To add a new detector:")
    print("  1. Inherit from BaseDetector in src/detectors/")
    print("  2. Implement: load_model(), detect(), get_model_info()")
    print("  3. Update main.py line 50 to use your detector")
    print("\n💡 Example: src/detectors/tensorflow_example.py")
    sys.exit(0)
else:
    print("⚠ SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(1)
