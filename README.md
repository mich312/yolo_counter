# yolo_counter

Count detections in images with yolo model. E.g. count people in images and store result in database.

**🎉 Now with modular architecture!** Easily swap detection models (YOLO, TensorFlow, PyTorch, etc.) without changing the core codebase.

## Features

- **Modular architecture** - Easy to extend and customize (see [ARCHITECTURE.md](ARCHITECTURE.md))
- **Pluggable detectors** - Swap YOLO with any detection model
- Object detection using YOLOv3 deep learning model
- Support for multiple webcam feeds
- OCR timestamp extraction from images
- PostgreSQL database storage for detection results
- Image segmentation for improved detection on large images
- Environment variable configuration for security
- Performance optimized (models loaded once, not per webcam)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual configuration:
```
POSTGRES_HOST=localhost
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database
POSTGRES_PORT=5433
YOLO_MODEL=yolov3
```

### 3. Setup PostgreSQL Database

Run the SQL schema script:

```bash
psql -h localhost -U your_username -d your_database -f setup.sql
```

### 4. Add Webcam URLs

Insert webcam URLs into the `webcam_urls` table:

```sql
INSERT INTO webcam_urls (url, name, debug)
VALUES ('http://example.com/webcam.jpg', 'Camera 1', false);
```

### 5. Download YOLO Weights

Download YOLOv3 weights and config from https://pjreddie.com/darknet/yolo/

Place the files in your model directory (default: current directory):
- `yolov3.weights`
- `yolov3.cfg`

Or use the tiny version for faster processing:
- `yolov3-tiny.weights`
- `yolov3-tiny.cfg`

### 6. Run the Application

```bash
python main.py
```

### 7. Schedule Repeated Execution

Use cron (Linux/Mac) or Task Scheduler (Windows) to run the script periodically.

Example cron job (runs every 5 minutes):
```
*/5 * * * * /usr/bin/python3 /path/to/yolo_counter/main.py
```

## Testing

Run the test suites to verify your setup:

```bash
# Test modular structure (no dependencies needed)
python3 test_structure.py

# Test code quality
python3 test_code_quality.py

# Full integration tests (requires dependencies)
python3 test_main.py
```

This validates:
- Python syntax
- Module structure and imports
- Configuration management
- Environment variable support
- Performance optimizations
- SQL schema correctness
- File structure

## Configuration

Configuration is now managed through environment variables (`.env` file):

**Database:**
- `POSTGRES_HOST` - Database host
- `POSTGRES_PORT` - Database port (default: 5433)
- `POSTGRES_USER` - Database username
- `POSTGRES_PASSWORD` - Database password
- `POSTGRES_DB` - Database name

**Detector:**
- `YOLO_MODEL` - Model name (yolov3 or yolov3-tiny)
- `CONFIDENCE_THRESHOLD` - Detection confidence (default: 0.5)
- `NMS_THRESHOLD` - Non-maximum suppression (default: 0.4)
- `TILE_SIZE` - Image tile size (default: 416)
- `DETECTION_CLASSES` - Classes to detect (default: person)

**OCR:**
- `OCR_ENABLED` - Enable OCR (default: true)
- `OCR_LANGUAGES` - Languages (default: de,en)

**Storage:**
- `IMAGE_FOLDER` - Where to save images (default: images.nosync)
- `DISK` - Optional disk path prefix
- `MODEL` - Model files directory

## Project Structure

```
yolo_counter/
├── main.py                  # Main application (orchestration)
├── setup.sql                # Database schema
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
├── .env                     # Your configuration (create this)
│
├── src/                     # Modular source code
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── database.py          # Database operations
│   ├── ocr.py               # Timestamp extraction
│   └── detectors/           # Detection models
│       ├── __init__.py
│       ├── base.py          # Abstract detector interface
│       ├── yolo.py          # YOLO implementation
│       └── tensorflow_example.py  # Template for new detectors
│
├── test_structure.py        # Structural tests (no deps)
├── test_code_quality.py     # Code quality tests
├── test_main.py             # Full integration tests
├── test_modular.py          # Modular architecture tests
│
├── ARCHITECTURE.md          # Architecture documentation
└── README.md                # This file
```

## Architecture

The project uses a **modular architecture** with clear separation of concerns:

- **Config Module** - Centralized configuration from environment variables
- **Database Module** - All database operations
- **OCR Module** - Timestamp extraction
- **Detectors Module** - Pluggable detection models
- **Main Script** - Orchestrates all components

### Adding a New Detector

Want to use a different detection model? It's easy!

1. Create a new detector class in `src/detectors/`
2. Inherit from `BaseDetector`
3. Implement: `load_model()`, `detect()`, `get_model_info()`
4. Update `main.py` line 50 to use your detector

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed instructions and examples.

### Example: Custom Detector

```python
from src.detectors.base import BaseDetector, DetectionResult

class MyDetector(BaseDetector):
    def load_model(self):
        # Load your model
        self.model = load_my_model()

    def detect(self, image):
        # Run detection
        detections = self.model.predict(image)
        return [DetectionResult(...) for d in detections]

    def get_model_info(self):
        return {'name': 'My Detector', 'version': '1.0'}
```

Then in `main.py`:
```python
from src.detectors.my_detector import MyDetector
self.detector = MyDetector(detector_config)
```

Done! Your custom model is now integrated.
