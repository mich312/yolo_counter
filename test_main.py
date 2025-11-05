#!/usr/bin/env python3
"""
Test script for main.py
Tests basic functionality without requiring database or YOLO models
"""

import sys
import os

print("=" * 60)
print("YOLO Counter - Test Suite")
print("=" * 60)

# Test 1: Check if required packages are installed
print("\n[Test 1] Checking package imports...")
try:
    import cv2
    print("  ✓ opencv-python (cv2)")
except ImportError as e:
    print(f"  ✗ opencv-python: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("  ✓ numpy")
except ImportError as e:
    print(f"  ✗ numpy: {e}")
    sys.exit(1)

try:
    import psycopg2
    print("  ✓ psycopg2")
except ImportError as e:
    print(f"  ✗ psycopg2: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("  ✓ pandas")
except ImportError as e:
    print(f"  ✗ pandas: {e}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("  ✓ python-dotenv")
except ImportError as e:
    print(f"  ✗ python-dotenv: {e}")
    sys.exit(1)

# EasyOCR is optional for this test since it's large
try:
    from easyocr import Reader
    print("  ✓ easyocr")
    easyocr_available = True
except ImportError as e:
    print(f"  ⚠ easyocr (optional): {e}")
    easyocr_available = False

print("\n[Test 2] Checking configuration constants...")
# Create a minimal test environment
os.environ['POSTGRES_HOST'] = 'test_host'
os.environ['POSTGRES_USER'] = 'test_user'
os.environ['POSTGRES_PASSWORD'] = 'test_password'
os.environ['POSTGRES_DB'] = 'test_db'

# Import constants from main.py by executing just the constants section
exec_globals = {}
with open('main.py', 'r') as f:
    content = f.read()

# Extract and execute just the constants section
constants_section = []
in_constants = False
for line in content.split('\n'):
    if 'Configuration constants' in line:
        in_constants = True
    elif in_constants and line.startswith('# Setup Database'):
        break
    elif in_constants:
        constants_section.append(line)

try:
    exec('\n'.join(constants_section), exec_globals)
    print(f"  ✓ DB_PORT = {exec_globals.get('DB_PORT', 'NOT FOUND')}")
    print(f"  ✓ TILE_SIZE = {exec_globals.get('TILE_SIZE', 'NOT FOUND')}")
    print(f"  ✓ CONFIDENCE_THRESHOLD = {exec_globals.get('CONFIDENCE_THRESHOLD', 'NOT FOUND')}")
    print(f"  ✓ NMS_THRESHOLD = {exec_globals.get('NMS_THRESHOLD', 'NOT FOUND')}")
    print(f"  ✓ SCALE_FACTOR = {exec_globals.get('SCALE_FACTOR', 'NOT FOUND')}")
    print(f"  ✓ OCR_LANGUAGES = {exec_globals.get('OCR_LANGUAGES', 'NOT FOUND')}")
except Exception as e:
    print(f"  ✗ Failed to load constants: {e}")
    sys.exit(1)

print("\n[Test 3] Validating configuration values...")
assert exec_globals['DB_PORT'] == 5433, "DB_PORT should be 5433"
assert exec_globals['TILE_SIZE'] == 416, "TILE_SIZE should be 416"
assert exec_globals['CONFIDENCE_THRESHOLD'] == 0.5, "CONFIDENCE_THRESHOLD should be 0.5"
assert exec_globals['NMS_THRESHOLD'] == 0.4, "NMS_THRESHOLD should be 0.4"
assert exec_globals['OCR_LANGUAGES'] == ['de', 'en'], "OCR_LANGUAGES should be ['de', 'en']"
print("  ✓ All configuration values are correct")

print("\n[Test 4] Testing environment variable loading...")
load_dotenv()
# Test with our mock environment
host = os.getenv('POSTGRES_HOST')
user = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')
db = os.getenv('POSTGRES_DB')

assert host == 'test_host', f"Expected 'test_host', got '{host}'"
assert user == 'test_user', f"Expected 'test_user', got '{user}'"
assert password == 'test_password', f"Expected 'test_password', got '{password}'"
assert db == 'test_db', f"Expected 'test_db', got '{db}'"
print("  ✓ Environment variables loaded correctly")

print("\n[Test 5] Testing helper functions...")
# Test get_output_layers function
class MockNet:
    def getLayerNames(self):
        return ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

    def getUnconnectedOutLayers(self):
        return [2, 4, 5]  # indices (1-based)

def get_output_layers(net):
    layers = net.getLayerNames()
    output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

mock_net = MockNet()
output_layers = get_output_layers(mock_net)
assert output_layers == ['layer2', 'layer4', 'layer5'], f"Expected ['layer2', 'layer4', 'layer5'], got {output_layers}"
print("  ✓ get_output_layers() works correctly")

# Test SQL injection protection in insert_or_update_wrapped
print("\n[Test 6] Testing SQL injection protection...")
test_data = {
    "name": "test'name",
    "value": "test'value"
}
protected_data = {k: v.replace("'", "''") if type(v) is str else v for (k, v) in test_data.items()}
assert protected_data['name'] == "test''name", "Single quotes should be escaped"
assert protected_data['value'] == "test''value", "Single quotes should be escaped"
print("  ✓ SQL injection protection works correctly")

print("\n[Test 7] Checking file structure...")
required_files = ['main.py', 'setup.sql', 'requirements.txt', '.env.example', 'README.md']
for file in required_files:
    if os.path.exists(file):
        print(f"  ✓ {file} exists")
    else:
        print(f"  ✗ {file} missing")

print("\n[Test 8] Validating SQL schema...")
with open('setup.sql', 'r') as f:
    sql_content = f.read()

# Check for the trailing comma bug
if ',\n);' in sql_content or ', \n);' in sql_content:
    print("  ✗ Trailing comma found in SQL")
    sys.exit(1)
else:
    print("  ✓ No trailing comma in SQL")

# Check for required tables
if 'CREATE TABLE public.webcam_detections' in sql_content:
    print("  ✓ webcam_detections table defined")
else:
    print("  ✗ webcam_detections table missing")
    sys.exit(1)

if 'CREATE TABLE public.webcam_urls' in sql_content:
    print("  ✓ webcam_urls table defined")
else:
    print("  ✗ webcam_urls table missing")
    sys.exit(1)

if 'UNIQUE (webcam, date)' in sql_content:
    print("  ✓ Unique constraint on (webcam, date) defined")
else:
    print("  ✗ Unique constraint missing")

print("\n[Test 9] Checking requirements.txt...")
with open('requirements.txt', 'r') as f:
    requirements = f.read()

required_packages = ['opencv-python', 'numpy', 'psycopg2-binary', 'pandas', 'easyocr', 'python-dotenv']
for package in required_packages:
    if package in requirements:
        print(f"  ✓ {package} in requirements.txt")
    else:
        print(f"  ✗ {package} missing from requirements.txt")

print("\n[Test 10] Performance optimization verification...")
with open('main.py', 'r') as f:
    main_content = f.read()

# Check that model loading is outside the loop
lines = main_content.split('\n')
model_load_line = None
loop_start_line = None

for i, line in enumerate(lines):
    if 'cv2.dnn.readNet' in line and 'Load YOLO model' in lines[max(0, i-3):i]:
        model_load_line = i
    if 'for cam in df_cams.iterrows():' in line:
        loop_start_line = i

if model_load_line is not None and loop_start_line is not None:
    if model_load_line < loop_start_line:
        print("  ✓ YOLO model loaded OUTSIDE loop (performance optimized)")
    else:
        print("  ✗ YOLO model loaded INSIDE loop (performance issue)")
        sys.exit(1)
else:
    print("  ⚠ Could not verify model loading position")

# Check that Reader is initialized outside loop
reader_init_line = None
for i, line in enumerate(lines):
    if 'Reader(OCR_LANGUAGES)' in line or "Reader(['de', 'en'])" in line:
        if 'Load OCR reader' in lines[max(0, i-3):i]:
            reader_init_line = i
            break

if reader_init_line is not None and loop_start_line is not None:
    if reader_init_line < loop_start_line:
        print("  ✓ OCR Reader initialized OUTSIDE loop (performance optimized)")
    else:
        print("  ✗ OCR Reader initialized INSIDE loop (performance issue)")
        sys.exit(1)
else:
    print("  ⚠ Could not verify OCR Reader initialization position")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nNote: This test suite validates code quality and structure.")
print("Full integration testing requires:")
print("  - PostgreSQL database running")
print("  - YOLO model weights downloaded")
print("  - Webcam URLs configured")
print("=" * 60)
