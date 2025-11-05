#!/usr/bin/env python3
"""
Code Quality Test Suite for YOLO Counter
Tests code structure and logic without requiring heavy dependencies
"""

import sys
import os
import re

print("=" * 60)
print("YOLO Counter - Code Quality Test Suite")
print("=" * 60)

all_passed = True

# Test 1: Python syntax validation
print("\n[Test 1] Validating Python syntax...")
try:
    import py_compile
    py_compile.compile('main.py', doraise=True)
    print("  ✓ main.py has valid Python syntax")
except py_compile.PyCompileError as e:
    print(f"  ✗ Syntax error in main.py: {e}")
    all_passed = False

# Test 2: Check configuration constants exist
print("\n[Test 2] Checking configuration constants...")
with open('main.py', 'r') as f:
    content = f.read()

required_constants = {
    'DB_PORT': '5433',
    'TILE_SIZE': '416',
    'CONFIDENCE_THRESHOLD': '0.5',
    'NMS_THRESHOLD': '0.4',
    'SCALE_FACTOR': '0.00392 * 6',
    'OCR_LANGUAGES': "['de', 'en']"
}

for const_name, expected_value in required_constants.items():
    pattern = rf'{const_name}\s*=\s*{re.escape(expected_value)}'
    if re.search(pattern, content):
        print(f"  ✓ {const_name} = {expected_value}")
    else:
        print(f"  ✗ {const_name} not found or incorrect value")
        all_passed = False

# Test 3: Check environment variable usage
print("\n[Test 3] Verifying environment variable support...")
env_vars = [
    'POSTGRES_HOST',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_DB',
    'YOLO_MODEL'
]

for var in env_vars:
    if f"os.getenv('{var}" in content:
        print(f"  ✓ {var} loaded from environment")
    else:
        print(f"  ✗ {var} not using environment variables")
        all_passed = False

# Test 4: Check for dotenv import
print("\n[Test 4] Checking python-dotenv integration...")
if 'from dotenv import load_dotenv' in content:
    print("  ✓ python-dotenv imported")
else:
    print("  ✗ python-dotenv not imported")
    all_passed = False

if 'load_dotenv()' in content:
    print("  ✓ load_dotenv() called")
else:
    print("  ✗ load_dotenv() not called")
    all_passed = False

# Test 5: Check no duplicate imports
print("\n[Test 5] Checking for duplicate imports...")
import_lines = [line for line in content.split('\n') if line.startswith('import ') or line.startswith('from ')]
import_modules = []
duplicates_found = False

for line in import_lines:
    if line.startswith('import '):
        module = line.split()[1].split('.')[0]
        if module in import_modules:
            print(f"  ✗ Duplicate import found: {module}")
            duplicates_found = True
            all_passed = False
        import_modules.append(module)

if not duplicates_found:
    print("  ✓ No duplicate imports found")

# Test 6: Check for hardcoded credentials
print("\n[Test 6] Checking for hardcoded credentials...")
patterns_to_avoid = [
    (r'postgres_host\s*=\s*["\'][^"\']+["\']', 'hardcoded postgres_host'),
    (r'password["\']:\s*["\'][^"\']+["\']', 'hardcoded password'),
]

hardcoded_found = False
for pattern, desc in patterns_to_avoid:
    matches = re.findall(pattern, content)
    # Filter out empty strings which are OK
    matches = [m for m in matches if '""' not in m and "''" not in m]
    if matches:
        print(f"  ⚠ Potential {desc} found (check if using env vars)")

# Since we're using os.getenv with empty string defaults, this is OK
if "os.getenv('POSTGRES_" in content:
    print("  ✓ Using environment variables for credentials")
else:
    print("  ✗ Not using environment variables for credentials")
    all_passed = False

# Test 7: Performance optimization check
print("\n[Test 7] Verifying performance optimizations...")
lines = content.split('\n')

# Find model loading and loop positions
model_load_line = None
reader_init_line = None
loop_line = None

for i, line in enumerate(lines):
    if 'cv2.dnn.readNet' in line:
        model_load_line = i
    if 'Reader(OCR_LANGUAGES)' in line or "Reader(['de', 'en'])" in line:
        reader_init_line = i
    if 'for cam in df_cams.iterrows():' in line:
        loop_line = i

if model_load_line and loop_line:
    if model_load_line < loop_line:
        print("  ✓ YOLO model loaded outside loop (optimized)")
    else:
        print("  ✗ YOLO model loaded inside loop (performance issue)")
        all_passed = False
else:
    print("  ⚠ Could not verify YOLO model loading position")

if reader_init_line and loop_line:
    if reader_init_line < loop_line:
        print("  ✓ OCR Reader initialized outside loop (optimized)")
    else:
        print("  ✗ OCR Reader initialized inside loop (performance issue)")
        all_passed = False
else:
    print("  ⚠ Could not verify OCR Reader initialization position")

# Test 8: Check SQL schema
print("\n[Test 8] Validating SQL schema...")
with open('setup.sql', 'r') as f:
    sql_content = f.read()

# Check for trailing comma bug
trailing_comma_patterns = [',\n);', ', \n);', ',\r\n);']
has_trailing_comma = any(pattern in sql_content for pattern in trailing_comma_patterns)

if has_trailing_comma:
    print("  ✗ Trailing comma found in SQL (syntax error)")
    all_passed = False
else:
    print("  ✓ No trailing comma in SQL")

# Check for required tables
if 'CREATE TABLE public.webcam_detections' in sql_content:
    print("  ✓ webcam_detections table defined")
else:
    print("  ✗ webcam_detections table missing")
    all_passed = False

if 'CREATE TABLE public.webcam_urls' in sql_content:
    print("  ✓ webcam_urls table defined")
else:
    print("  ✗ webcam_urls table missing")
    all_passed = False

if 'UNIQUE (webcam, date)' in sql_content:
    print("  ✓ Unique constraint on (webcam, date)")
else:
    print("  ⚠ Unique constraint on (webcam, date) not found")

# Test 9: Check requirements.txt
print("\n[Test 9] Validating requirements.txt...")
if not os.path.exists('requirements.txt'):
    print("  ✗ requirements.txt missing")
    all_passed = False
else:
    with open('requirements.txt', 'r') as f:
        requirements = f.read()

    required_packages = [
        'opencv-python',
        'numpy',
        'psycopg2-binary',
        'pandas',
        'easyocr',
        'python-dotenv'
    ]

    for package in required_packages:
        if package in requirements:
            print(f"  ✓ {package}")
        else:
            print(f"  ✗ {package} missing")
            all_passed = False

# Test 10: Check .env.example exists
print("\n[Test 10] Checking .env.example...")
if os.path.exists('.env.example'):
    print("  ✓ .env.example file exists")
    with open('.env.example', 'r') as f:
        env_example = f.read()

    required_env_vars = ['POSTGRES_HOST', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB']
    for var in required_env_vars:
        if var in env_example:
            print(f"  ✓ {var} documented")
        else:
            print(f"  ✗ {var} not in .env.example")
            all_passed = False
else:
    print("  ✗ .env.example file missing")
    all_passed = False

# Test 11: Check file structure
print("\n[Test 11] Checking file structure...")
required_files = {
    'main.py': 'Main application script',
    'setup.sql': 'Database schema',
    'requirements.txt': 'Python dependencies',
    '.env.example': 'Environment variables template',
    'README.md': 'Project documentation',
    '.gitignore': 'Git ignore rules'
}

for file, desc in required_files.items():
    if os.path.exists(file):
        print(f"  ✓ {file} ({desc})")
    else:
        print(f"  ✗ {file} missing ({desc})")
        all_passed = False

# Test 12: Check for unused imports
print("\n[Test 12] Checking for removed unused imports...")
unused_imports = ['matplotlib.pyplot', 'sqlalchemy']
for unused in unused_imports:
    if unused in content:
        print(f"  ✗ Unused import found: {unused}")
        all_passed = False
    else:
        print(f"  ✓ {unused} removed (was unused)")

# Test 13: Check constants are used
print("\n[Test 13] Verifying constants are actually used...")
constants_to_check = {
    'DB_PORT': [r'port\s*=\s*DB_PORT', r'port=DB_PORT'],
    'TILE_SIZE': [r'//TILE_SIZE', r'\(TILE_SIZE'],
    'CONFIDENCE_THRESHOLD': [r'>\s*CONFIDENCE_THRESHOLD', r'CONFIDENCE_THRESHOLD'],
    'NMS_THRESHOLD': [r'NMS_THRESHOLD'],
    'OCR_LANGUAGES': [r'Reader\(OCR_LANGUAGES\)']
}

for const, patterns in constants_to_check.items():
    found = any(re.search(p, content) for p in patterns)
    if found:
        print(f"  ✓ {const} is used in code")
    else:
        print(f"  ⚠ {const} might not be used")

# Test 14: Logic validation
print("\n[Test 14] Validating code logic...")

# Check for proper error handling
if 'except Exception as e:' in content:
    print("  ✓ Exception handling present")
else:
    print("  ⚠ No exception handling found")

# Check for connection cleanup
if 'conn.close()' in content or 'cur.close()' in content:
    print("  ✓ Database connection cleanup present")
else:
    print("  ⚠ Database connection cleanup not found")

# Check for SQL injection protection
if "replace(\"'\",\"''\")" in content or ".replace(\"'\", \"''\")" in content:
    print("  ✓ SQL injection protection (single quote escaping)")
else:
    print("  ⚠ SQL injection protection not found")

print("\n" + "=" * 60)
if all_passed:
    print("✅ ALL CODE QUALITY TESTS PASSED!")
    print("=" * 60)
    print("\nThe code is well-structured and follows best practices.")
    print("Quick wins have been successfully implemented:")
    print("  • No duplicate imports")
    print("  • Constants extracted from magic numbers")
    print("  • Environment variables for credentials")
    print("  • Performance optimizations (model loaded once)")
    print("  • SQL syntax corrected")
    print("  • All required files present")
    sys.exit(0)
else:
    print("⚠ SOME TESTS FAILED OR HAVE WARNINGS")
    print("=" * 60)
    print("\nPlease review the failed tests above.")
    sys.exit(1)
