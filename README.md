# yolo_counter
Count detections in images with yolo model. E.g. count people in images and store result in database.

## Setup

1. Setup PostgreSQL database by using `setup.sql` script
2. Fill in Database credentials inside the `main.py` script.
3. Add Webcam URLs to `webcam_urls` table.
4. Download yolo weights and config from https://pjreddie.com/darknet/yolo/
5. Schedule script to run repeatedly.
