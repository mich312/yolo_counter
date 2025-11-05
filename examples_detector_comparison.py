#!/usr/bin/env python3
"""
Detector Comparison Example

Shows how to use different detection models with the modular architecture:
- YOLOv3 (OpenCV DNN - baseline)
- YOLO11 (Ultralytics - latest, recommended)
- RT-DETR (Ultralytics - transformer-based)

Run this to see how easy it is to swap detection models!
"""

import sys

print("=" * 70)
print("YOLO Counter - Detector Comparison Example")
print("=" * 70)

# Check what detectors are available
print("\n[Step 1] Checking available detectors...")

try:
    from src.detectors import MODERN_DETECTORS_AVAILABLE
    if MODERN_DETECTORS_AVAILABLE:
        print("✓ Modern detectors (YOLO11, RT-DETR) available")
    else:
        print("⚠ Modern detectors not available (ultralytics not installed)")
        print("  Install with: pip install ultralytics>=8.3.0")
except ImportError:
    print("ℹ️  Running in documentation mode (dependencies not installed)")
    print("  This script shows examples of how to use different detectors")
    MODERN_DETECTORS_AVAILABLE = False

print("\n" + "=" * 70)
print("Detector Usage Examples")
print("=" * 70)

# Example 1: YOLOv3 (OpenCV DNN)
print("\n📦 Example 1: YOLOv3 via OpenCV DNN (Baseline)")
print("-" * 70)
print("""
from src.detectors.yolo import YOLODetector

config = {
    'model_path': './yolov3.weights',
    'config_path': './yolov3.cfg',
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'tile_size': 416
}

detector = YOLODetector(config)
detector.load_model()

# Detect objects
results = detector.detect(image)

# Filter for people/skiers
people = detector.filter_by_class(results, ['person'])
print(f"Found {len(people)} people")

# Get counts
counts = detector.count_by_class(results)
print(f"Detections: {counts}")
""")

# Example 2: YOLO11
print("\n🚀 Example 2: YOLO11 via Ultralytics (Recommended)")
print("-" * 70)
print("""
from src.detectors.yolo11 import YOLO11Detector, create_yolo11_detector

# Method 1: Full configuration
config = {
    'model_name': 'yolo11n.pt',  # n=nano, s=small, m=medium, l=large, x=extra
    'confidence_threshold': 0.5,
    'device': 'cpu'  # or 'cuda' for GPU
}
detector = YOLO11Detector(config)

# Method 2: Quick creation (easier!)
detector = create_yolo11_detector(
    variant='n',        # Model size: n, s, m, l, x
    confidence=0.5,
    device='cpu'
)

detector.load_model()  # Auto-downloads model if needed

# Detect objects
results = detector.detect(image)

# YOLO11 has 80 COCO classes including 'person'
people = detector.filter_by_class(results, ['person'])
print(f"Found {len(people)} people/skiers")

# Model info
info = detector.get_model_info()
print(f"Using: {info['name']} ({info['release_date']})")
print(f"Best for: {info['best_for']}")
""")

# Example 3: RT-DETR
print("\n🔬 Example 3: RT-DETR via Ultralytics (Transformer-based)")
print("-" * 70)
print("""
from src.detectors.rtdetr import RTDETRDetector, create_rtdetr_detector

# Method 1: Full configuration
config = {
    'model_name': 'rtdetr-l.pt',  # l=large, x=extra-large
    'confidence_threshold': 0.5,
    'device': 'cpu'
}
detector = RTDETRDetector(config)

# Method 2: Quick creation
detector = create_rtdetr_detector(
    variant='l',        # Model size: l, x
    confidence=0.5,
    device='cpu'
)

detector.load_model()  # Auto-downloads model

# Detect objects (no NMS needed - it's end-to-end!)
results = detector.detect(image)

# Same interface as other detectors
people = detector.filter_by_class(results, ['person'])

# Model info
info = detector.get_model_info()
print(f"Architecture: {info['architecture']}")
print(f"Key features: {info['key_features']}")
""")

# Example 4: Side-by-side comparison
print("\n⚖️  Example 4: Side-by-Side Comparison")
print("-" * 70)
print("""
import time

# Load all three detectors
yolo3 = YOLODetector(yolo3_config)
yolo11 = create_yolo11_detector('n', confidence=0.5)
rtdetr = create_rtdetr_detector('l', confidence=0.5)

yolo3.load_model()
yolo11.load_model()
rtdetr.load_model()

detectors = {
    'YOLOv3 (OpenCV)': yolo3,
    'YOLO11 (Latest)': yolo11,
    'RT-DETR (Transformer)': rtdetr
}

# Compare on same image
for name, detector in detectors.items():
    start = time.time()
    results = detector.detect(image)
    elapsed = time.time() - start

    people = detector.filter_by_class(results, ['person'])

    print(f"\\n{name}:")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  People detected: {len(people)}")
    print(f"  Total detections: {len(results)}")
""")

# Example 5: Using in main.py
print("\n🔧 Example 5: Integrating into main.py")
print("-" * 70)
print("""
# In main.py, line 41-51, replace YOLODetector with YOLO11Detector:

# OLD (YOLOv3):
from src.detectors.yolo import YOLODetector
self.detector = YOLODetector(detector_config)

# NEW (YOLO11 - Recommended):
from src.detectors.yolo11 import YOLO11Detector
detector_config = {
    'model_name': 'yolo11n.pt',
    'confidence_threshold': config.detector.confidence_threshold,
    'device': 'cpu'  # or 'cuda'
}
self.detector = YOLO11Detector(detector_config)

# OR RT-DETR (Alternative):
from src.detectors.rtdetr import RTDETRDetector
detector_config = {
    'model_name': 'rtdetr-l.pt',
    'confidence_threshold': config.detector.confidence_threshold,
    'device': 'cpu'
}
self.detector = RTDETRDetector(detector_config)

# That's it! Everything else works the same!
""")

# Model comparison table
print("\n📊 Model Comparison Table")
print("-" * 70)
print("""
Model      | Speed | Accuracy | Parameters | Ultralytics | Best For
-----------|-------|----------|------------|-------------|------------------
YOLOv3     | 23ms  | Good     | High       | No (OpenCV) | Baseline, offline
YOLO11     | 13ms  | Best     | Low (-22%) | Yes         | ⭐ General use
RT-DETR    | 9ms*  | High     | Medium     | Yes         | Multi-object scenes

*GPU-dependent (108 FPS on T4 GPU)

Recommendation:
- Start with YOLO11 (best overall)
- Try RT-DETR if you have complex scenes
- Keep YOLOv3 if you can't install ultralytics
""")

# Installation help
print("\n💾 Installation")
print("-" * 70)
print("""
# Install all dependencies including YOLO11 and RT-DETR:
pip install -r requirements.txt

# Or install just ultralytics:
pip install ultralytics>=8.3.0

# Models auto-download on first use:
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # Downloads automatically!
""")

# Links
print("\n🔗 Learn More")
print("-" * 70)
print("""
- RESEARCH_MODELS.md - Comprehensive research on all models
- ARCHITECTURE.md - How the modular system works
- src/detectors/yolo11.py - YOLO11 implementation
- src/detectors/rtdetr.py - RT-DETR implementation
""")

print("\n" + "=" * 70)
print("✅ Examples complete!")
print("=" * 70)
print("""
Quick Start:
1. pip install ultralytics>=8.3.0
2. Update main.py line 41-51 to use YOLO11Detector
3. Run: python main.py

Your modular architecture makes swapping models trivial! 🚀
""")
