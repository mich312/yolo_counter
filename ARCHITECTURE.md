# YOLO Counter - Modular Architecture

## Overview

The YOLO Counter has been refactored into a modular architecture that separates concerns and makes it easy to swap detection models. The codebase is now organized into clearly defined modules with single responsibilities.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                          main.py                            │
│                    (Orchestration Layer)                    │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │         WebcamProcessor                            │   │
│  │  - Coordinates all components                      │   │
│  │  - Processes webcam feeds                          │   │
│  │  - Stores results                                  │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Config     │   │   Database   │   │     OCR      │
│   Module     │   │   Module     │   │   Module     │
│              │   │              │   │              │
│ - AppConfig  │   │ - Database   │   │ - Timestamp  │
│ - DB Config  │   │   Manager    │   │   Extractor  │
│ - Detector   │   │ - Query      │   │ - EasyOCR    │
│   Config     │   │ - Insert     │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │  Detectors   │
                   │   Module     │
                   │              │
                   │ ┌──────────┐ │
                   │ │   Base   │ │ ← Abstract interface
                   │ │ Detector │ │
                   │ └──────────┘ │
                   │      △       │
                   │      │       │
                   │ ┌────┴────┐  │
                   │ │  YOLO   │  │ ← Implementation
                   │ │Detector │  │
                   │ └─────────┘  │
                   │              │
                   │ ┌──────────┐ │
                   │ │TensorFlow│ │ ← Example template
                   │ │ Example  │ │
                   │ └──────────┘ │
                   └──────────────┘
```

## Module Structure

### 1. Configuration Module (`src/config.py`)

**Purpose:** Centralized configuration management

**Components:**
- `DatabaseConfig` - PostgreSQL connection settings
- `DetectorConfig` - Model configuration (paths, thresholds)
- `OCRConfig` - OCR settings (languages, enabled)
- `StorageConfig` - File storage paths
- `AppConfig` - Master configuration container

**Features:**
- Loads from environment variables (`.env`)
- Type-safe with dataclasses
- Validation for required fields
- Factory method pattern (`from_env()`)

**Example:**
```python
from src.config import AppConfig

config = AppConfig.from_env()
print(config.detector.model_name)  # 'yolov3'
print(config.database.host)         # 'localhost'
```

### 2. Database Module (`src/database.py`)

**Purpose:** All PostgreSQL database operations

**Components:**
- `DatabaseManager` - Main database interface

**Methods:**
- `query(sql)` - Execute SELECT queries
- `get_active_webcams()` - Fetch webcam URLs
- `insert_or_update()` - UPSERT operations
- `store_detections()` - Save detection results

**Features:**
- Connection pooling
- Automatic SQL injection protection (quote escaping)
- Pandas DataFrame integration
- Timezone handling (Berlin → UTC conversion)

**Example:**
```python
from src.database import DatabaseManager
from src.config import DatabaseConfig

config = DatabaseConfig.from_env()
db = DatabaseManager(config)

webcams = db.get_active_webcams()
print(f"Found {len(webcams)} webcams")
```

### 3. OCR Module (`src/ocr.py`)

**Purpose:** Timestamp extraction from images

**Components:**
- `TimestampExtractor` - OCR-based timestamp parsing

**Methods:**
- `load_reader()` - Initialize EasyOCR (slow, call once)
- `extract_timestamp()` - Parse datetime from image

**Features:**
- Multi-language support (German, English by default)
- Regex-based date/time extraction
- Multiple date format parsing
- Graceful fallback on errors

**Example:**
```python
from src.ocr import TimestampExtractor

ocr = TimestampExtractor(['de', 'en'])
ocr.load_reader()

timestamp = ocr.extract_timestamp(image)
print(f"Extracted: {timestamp}")
```

### 4. Detectors Module (`src/detectors/`)

**Purpose:** Pluggable object detection models

#### Base Detector (`src/detectors/base.py`)

**Abstract interface for all detection models.**

**Key Classes:**
- `DetectionResult` - Dataclass for detection results
  - `class_id` - Numeric class identifier
  - `class_name` - Human-readable class name
  - `confidence` - Detection confidence (0.0-1.0)
  - `bbox` - Bounding box (x, y, width, height)

- `BaseDetector` - Abstract base class
  - `load_model()` - Load model weights (abstract)
  - `detect(image)` - Perform detection (abstract)
  - `get_model_info()` - Return model metadata (abstract)
  - `filter_by_class()` - Filter results by class names
  - `filter_by_confidence()` - Filter by threshold
  - `count_by_class()` - Count detections per class

**Example:**
```python
from src.detectors.base import BaseDetector, DetectionResult

class MyDetector(BaseDetector):
    def load_model(self):
        # Load your model here
        pass

    def detect(self, image):
        # Run inference
        results = []
        # ... detection logic ...
        return results

    def get_model_info(self):
        return {'name': 'My Model', 'version': '1.0'}
```

#### YOLO Detector (`src/detectors/yolo.py`)

**YOLOv3 implementation using OpenCV DNN.**

**Features:**
- Pre-configured with 80 COCO classes
- Image segmentation for large images
- Non-Maximum Suppression (NMS)
- Bounding box visualization

**Configuration:**
```python
detector_config = {
    'model_path': '/path/to/yolov3.weights',
    'config_path': '/path/to/yolov3.cfg',
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'tile_size': 416,
    'scale_factor': 0.00392 * 6
}

detector = YOLODetector(detector_config)
detector.load_model()

results = detector.detect(image)
for result in results:
    print(f"{result.class_name}: {result.confidence:.2f}")
```

#### TensorFlow Example (`src/detectors/tensorflow_example.py`)

**Template for adding TensorFlow models.**

This file shows the structure needed to add a new detector. Simply:
1. Inherit from `BaseDetector`
2. Implement the three abstract methods
3. Update `main.py` to use your detector

## Main Application (`main.py`)

**Purpose:** Orchestration and workflow coordination

**Components:**
- `WebcamProcessor` - Main processing class
  - Initializes all components
  - Processes webcam feeds
  - Coordinates detection pipeline
  - Stores results

**Workflow:**
```
1. Load configuration from .env
2. Initialize components:
   - DatabaseManager
   - Detector (YOLO by default)
   - TimestampExtractor (if enabled)
3. Fetch active webcams from database
4. For each webcam:
   a. Fetch image
   b. Extract timestamp (OCR)
   c. Run detection
   d. Filter by configured classes
   e. Save images (raw + annotated)
   f. Store results in database
5. Print summary
```

## How to Add a New Detector

### Step 1: Create Detector Class

Create a new file in `src/detectors/`, e.g., `my_detector.py`:

```python
from typing import List
import numpy as np
from .base import BaseDetector, DetectionResult

class MyDetector(BaseDetector):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_path = config['model_path']
        # Add your configuration

    def load_model(self):
        """Load your model"""
        print("Loading my model...")
        # Load model weights/architecture
        self.model = load_my_model(self.model_path)
        self.classes = ['person', 'car', ...]  # Your classes

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Run detection"""
        # Run your model
        raw_detections = self.model.predict(image)

        # Convert to DetectionResult format
        results = []
        for det in raw_detections:
            result = DetectionResult(
                class_id=det['class_id'],
                class_name=self.classes[det['class_id']],
                confidence=det['score'],
                bbox=(det['x'], det['y'], det['w'], det['h'])
            )
            results.append(result)

        return results

    def get_model_info(self) -> dict:
        """Return model metadata"""
        return {
            'name': 'My Custom Detector',
            'framework': 'MyFramework',
            'model_path': self.model_path,
            'num_classes': len(self.classes)
        }
```

### Step 2: Update Main.py

Modify `main.py` line 50 to use your detector:

```python
# OLD:
from src.detectors.yolo import YOLODetector
self.detector = YOLODetector(detector_config)

# NEW:
from src.detectors.my_detector import MyDetector
self.detector = MyDetector(detector_config)
```

### Step 3: Configure

Add any new configuration to `.env`:

```bash
DETECTOR_TYPE=my_detector
MY_MODEL_PATH=/path/to/model.pth
```

### Step 4: Run

```bash
python main.py
```

That's it! Your detector is now integrated.

## Benefits of Modular Architecture

### 1. Separation of Concerns
Each module has a single, well-defined responsibility:
- Config: Configuration management
- Database: Data persistence
- OCR: Timestamp extraction
- Detectors: Object detection
- Main: Orchestration

### 2. Testability
Each module can be tested independently:
```python
# Test detector without database
detector = YOLODetector(config)
results = detector.detect(test_image)
assert len(results) > 0

# Test database without detector
db = DatabaseManager(config)
webcams = db.get_active_webcams()
assert len(webcams) > 0
```

### 3. Maintainability
- Clear file organization
- Type hints throughout
- Comprehensive docstrings
- Logical grouping of related code

### 4. Extensibility
- Easy to add new detectors (inherit from BaseDetector)
- Easy to add new storage backends (inherit from DatabaseManager)
- Easy to add new OCR engines (inherit from TimestampExtractor)

### 5. Reusability
Modules can be used in other projects:
```python
# Use just the detector
from src.detectors.yolo import YOLODetector
detector = YOLODetector(config)
results = detector.detect(my_image)

# Use just the config
from src.config import AppConfig
config = AppConfig.from_env()
```

## Code Metrics

- **Old main.py:** 277 lines (monolithic)
- **New main.py:** 248 lines (orchestration only)
- **Total modular code:** ~1100 lines across 9 files
- **Reduction in complexity:** 10.5%
- **Number of modules:** 4 main modules + detectors package

## Testing

Run structural tests:
```bash
python3 test_structure.py
```

Run code quality tests:
```bash
python3 test_code_quality.py
```

Run full integration tests (requires dependencies):
```bash
python3 test_main.py
```

## Migration Guide

### From Old Version

If you were using the old monolithic `main.py`:

1. **Configuration**: Move credentials from code to `.env`
2. **No code changes needed**: The new version is backwards compatible
3. **Optional**: If you want to customize, see "How to Add a New Detector"

### Breaking Changes

None! The new architecture is a drop-in replacement.

## Future Enhancements

Potential additions to the modular architecture:

1. **Storage Backends**
   - S3 storage instead of local filesystem
   - Different database types (MySQL, MongoDB)

2. **More Detectors**
   - YOLOv5/v8 support
   - TensorFlow Object Detection API
   - PyTorch detectors (Detectron2, MMDetection)

3. **Preprocessing**
   - Image enhancement module
   - Camera calibration
   - Motion detection

4. **Post-processing**
   - Object tracking across frames
   - Activity recognition
   - Alert/notification system

5. **API Layer**
   - REST API for real-time detection
   - WebSocket streaming
   - Web dashboard

All of these can be added without modifying existing code, thanks to the modular design!
