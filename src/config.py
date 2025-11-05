"""
Configuration management

Loads configuration from environment variables and provides structured access
"""

import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    username: str
    password: str
    database: str

    @classmethod
    def from_env(cls):
        """Create from environment variables"""
        return cls(
            host=os.getenv('POSTGRES_HOST', ''),
            port=int(os.getenv('POSTGRES_PORT', '5433')),
            username=os.getenv('POSTGRES_USER', ''),
            password=os.getenv('POSTGRES_PASSWORD', ''),
            database=os.getenv('POSTGRES_DB', '')
        )

    def validate(self):
        """Validate that all required fields are set"""
        if not all([self.host, self.username, self.password, self.database]):
            raise ValueError(
                "Missing required database configuration. "
                "Please set POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB"
            )


@dataclass
class DetectorConfig:
    """Detector configuration"""
    model_name: str
    model_path: str
    config_path: str
    confidence_threshold: float
    nms_threshold: float
    tile_size: int
    scale_factor: float
    detection_classes: List[str]

    @classmethod
    def from_env(cls):
        """Create from environment variables"""
        model_name = os.getenv('YOLO_MODEL', 'yolov3')
        model_folder = os.getenv('MODEL', '.')
        model_folder = f"{model_folder}/yolo" if model_folder != '.' else model_folder

        return cls(
            model_name=model_name,
            model_path=f"{model_folder}/{model_name}.weights",
            config_path=f"{model_folder}/{model_name}.cfg",
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
            nms_threshold=float(os.getenv('NMS_THRESHOLD', '0.4')),
            tile_size=int(os.getenv('TILE_SIZE', '416')),
            scale_factor=float(os.getenv('SCALE_FACTOR', str(0.00392 * 6))),
            detection_classes=os.getenv('DETECTION_CLASSES', 'person').split(',')
        )


@dataclass
class OCRConfig:
    """OCR configuration"""
    languages: List[str]
    enabled: bool
    engine: str  # 'paddle' (default, 3-5x faster) or 'easy' (legacy)

    @classmethod
    def from_env(cls):
        """Create from environment variables"""
        return cls(
            languages=os.getenv('OCR_LANGUAGES', 'de,en').split(','),
            enabled=os.getenv('OCR_ENABLED', 'true').lower() == 'true',
            engine=os.getenv('OCR_ENGINE', 'paddle').lower()  # paddle (default) or easy
        )


@dataclass
class StorageConfig:
    """Storage configuration"""
    image_folder: str

    @classmethod
    def from_env(cls):
        """Create from environment variables"""
        disk = os.getenv('DISK', '')
        if disk:
            image_folder = f"{disk}/yolo"
        else:
            image_folder = os.getenv('IMAGE_FOLDER', 'images.nosync')

        return cls(image_folder=image_folder)


@dataclass
class AppConfig:
    """Main application configuration"""
    database: DatabaseConfig
    detector: DetectorConfig
    ocr: OCRConfig
    storage: StorageConfig

    @classmethod
    def from_env(cls):
        """Create configuration from environment variables"""
        config = cls(
            database=DatabaseConfig.from_env(),
            detector=DetectorConfig.from_env(),
            ocr=OCRConfig.from_env(),
            storage=StorageConfig.from_env()
        )

        # Validate critical configuration
        config.database.validate()

        return config

    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database,
                'username': self.database.username,
                'password': '***'  # Don't log passwords
            },
            'detector': {
                'model_name': self.detector.model_name,
                'confidence_threshold': self.detector.confidence_threshold,
                'nms_threshold': self.detector.nms_threshold,
                'detection_classes': self.detector.detection_classes
            },
            'ocr': {
                'languages': self.ocr.languages,
                'enabled': self.ocr.enabled,
                'engine': self.ocr.engine
            },
            'storage': {
                'image_folder': self.storage.image_folder
            }
        }
