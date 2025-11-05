# yolo_counter

Count detections in images with yolo model. E.g. count people in images and store result in database.

## Features

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

Run the code quality test suite to verify your setup:

```bash
python3 test_code_quality.py
```

This validates:
- Python syntax
- Configuration constants
- Environment variable support
- Performance optimizations
- SQL schema correctness
- File structure

## Configuration

All configuration constants are defined at the top of `main.py`:

- `DB_PORT`: PostgreSQL port (default: 5433)
- `TILE_SIZE`: Image tile size for segmentation (default: 416)
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.5)
- `NMS_THRESHOLD`: Non-maximum suppression threshold (default: 0.4)
- `OCR_LANGUAGES`: Languages for OCR (default: ['de', 'en'])

## Project Structure

```
yolo_counter/
├── main.py              # Main application script
├── setup.sql            # Database schema
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── .env                 # Your configuration (create this)
├── test_code_quality.py # Code quality test suite
├── test_main.py         # Full integration tests
└── README.md            # This file
```
