# Object Detection Models Research - Skier Detection

## Executive Summary

Research Date: November 5, 2025
Focus: Latest object detection models with emphasis on skier/person detection

### Key Findings

**Best Models for Skier Detection (2025):**
1. **YOLO11** - Latest from Ultralytics (Oct 2024) - **RECOMMENDED**
2. **RT-DETR** - Transformer-based, real-time (Baidu)
3. **RF-DETR** - State-of-the-art from Roboflow (60+ AP on COCO)
4. **YOLOv8-Pose** - For detailed skier movement tracking

---

## Latest Object Detection Models (2025)

### 1. YOLO11 (October 2024) ⭐ RECOMMENDED

**Developer:** Ultralytics
**Status:** Latest production-ready model
**License:** AGPL-3.0 (Commercial license available)

#### Performance Metrics
- **Speed:** 13.5ms inference (fastest among YOLO family)
- **Accuracy:** Higher mAP than YOLOv8 across all model sizes
- **Parameters:** 22% fewer params than YOLOv8m while achieving higher accuracy
- **Real-world:** Best performance in traffic detection, small object detection

#### Key Features
- Improved backbone and neck architecture
- Better feature extraction
- Optimized for both CPU and GPU inference
- Substantially faster CPU inference than YOLOv8
- Better at detecting small, distant objects

#### Variants Available
- YOLO11n (Nano) - Ultra-lightweight
- YOLO11s (Small)
- YOLO11m (Medium)
- YOLO11l (Large)
- YOLO11x (Extra-large)

#### Use Cases
- **Ideal for skier detection:** Excellent at detecting people at varying distances
- Real-time webcam monitoring
- Multi-camera systems
- Edge deployment (optimized for CPUs)

**Integration:** ✅ Available in Ultralytics package

---

### 2. YOLOv9 (February 2024)

**Developer:** Wang et al.
**Status:** Production-ready

#### Performance Metrics
- **Accuracy:** Highest mAP of 0.935 in weed detection study
- **Speed:** 23ms inference time
- **Innovation:** Programmable Gradient Information (PGI)

#### Key Features
- Generalized Efficient Layer Aggregation Network (GELAN)
- Addresses information loss in deep neural networks
- Best accuracy in head-to-head comparisons
- Excellent for scenarios requiring maximum accuracy

#### Trade-offs
- Slower than YOLO11
- More parameters than YOLO11
- Best when accuracy > speed

**Integration:** ⚠️ Not officially in Ultralytics (requires custom implementation)

---

### 3. RT-DETR (2023) - Transformer-Based

**Developer:** Baidu
**Status:** Production-ready
**Paper:** "DETRs Beat YOLOs on Real-time Object Detection"

#### Performance Metrics
- **Speed:** 108 FPS on Nvidia T4 GPU
- **Accuracy:** 53.1% AP (v1), 55%+ AP (v2)
- **Architecture:** Transformer-based (end-to-end)

#### Key Features
- **No NMS required:** Eliminates Non-Maximum Suppression
- **Stable inference:** Speed remains constant regardless of object count
- **End-to-end:** Simpler architecture than traditional detectors
- **Transformer backbone:** Different approach from CNN-based YOLO

#### Advantages
- More parallelizable than YOLO
- Better for complex scenes with many objects
- Consistent performance across varying object densities

**Integration:** ✅ Available in Ultralytics (RT-DETR and RT-DETRv2)

---

### 4. RF-DETR (March 2025) - State-of-the-Art

**Developer:** Roboflow
**Status:** Cutting-edge research
**License:** Apache 2.0

#### Performance Metrics
- **Accuracy:** 60+ AP on COCO (SOTA)
- **Speed:** Real-time capable
- **Claim:** "Fastest and most accurate for its size"

#### Variants (July 2025)
- RF-DETR Nano
- RF-DETR Small
- RF-DETR Medium
- RF-DETR Large (128M parameters)

#### Key Features
- First real-time model > 60 AP on COCO
- Transformer-based architecture
- Designed for fine-tuning
- Instance segmentation support

#### Trade-offs
- ⚠️ **Not integrated in Ultralytics**
- Requires separate Roboflow rf-detr framework
- Newer, less battle-tested than YOLO

**Integration:** ❌ Requires custom implementation (Roboflow framework)

---

## Skier-Specific Detection Research

### SkiTB Dataset & Challenge (2024-2025)

**Source:** Multiple research papers (WACV 2024, SkiTB Challenge 2025)

#### Key Research Findings

**1. Best Models for Skier Tracking:**
- **ReID-SAM** (2025): F1-score of 0.870 on SkiTB dataset
  - Uses YOLOv11 + Kalman filtering
  - OSNet-based Re-ID for identity preservation
  - Addresses camera switches, occlusions, rapid motion

**2. Challenges in Ski Resort Detection:**
- Multiple pan-tilt-zoom cameras
- Camera movements and switches
- Occlusions (trees, other skiers, terrain)
- Scale variations (distant vs. close skiers)
- Rapid motion and motion blur
- Varying environmental conditions (snow, fog, lighting)

**3. Detection Approaches:**
- **Standard person detection works well** for skiers
- No specialized "skier class" needed
- Fine-tuning on ski resort datasets improves performance
- Multi-camera tracking requires Re-ID models

### Recommended Approach for Ski Resorts

**For Single Camera:**
- YOLO11 with "person" class detection
- Optional: YOLOv8-Pose for movement analysis

**For Multi-Camera Systems:**
- YOLO11 for detection
- OSNet or similar Re-ID model for tracking across cameras
- Kalman filtering for trajectory prediction

**For Advanced Analysis:**
- YOLOv11-Pose for keypoint detection (17 keypoints)
- Can track:
  - Skiing posture
  - Movement patterns
  - Injury risk assessment
  - Technique analysis

---

## Pose Estimation Models

### YOLOv8-Pose / YOLO11-Pose

**Use Case:** Detailed skier movement and posture tracking

#### Capabilities
- 17 keypoints per person (elbows, knees, head, etc.)
- 2D [x, y] or 3D [x, y, visible] coordinates
- Real-time performance

#### Recent Improvements (2025)
- **EE-YOLOv8:** Enhanced pose estimation
  - 89.0% AP at IoU 0.5 (+3.3% over baseline)
  - 65.6% AP over IoU 0.5-0.95 (+5.8%)

#### Sports Analytics Applications
- Athlete movement analysis
- Performance optimization
- Injury prevention
- Technique assessment

**Integration:** ✅ Available in Ultralytics (yolov8n-pose.pt, yolo11n-pose.pt)

---

## Model Comparison Matrix

| Model | Speed (ms) | Accuracy (mAP) | Parameters | Real-time | Ultralytics | Best For |
|-------|------------|----------------|------------|-----------|-------------|----------|
| **YOLO11** | **13.5** | **Highest** | **Low** | ✅ Yes | ✅ Yes | **General purpose, skiers** |
| YOLOv9 | 23 | Very High | Medium | ✅ Yes | ❌ No | Maximum accuracy |
| YOLOv8 | 23 | High | Medium | ✅ Yes | ✅ Yes | Proven, reliable |
| RT-DETR | 9.3* | High (53-55%) | Medium | ✅ Yes | ✅ Yes | Transformer approach |
| RF-DETR | Fast | **60+ AP** | Varies | ✅ Yes | ❌ No | Cutting-edge research |
| YOLO11-Pose | ~15 | High | Medium | ✅ Yes | ✅ Yes | Movement tracking |

*GPU-dependent; 108 FPS on T4

---

## Recommendations

### For Your Webcam Counter Project

#### Primary Detector: YOLO11 ⭐
**Why:**
- Latest and most optimized (Oct 2024)
- 22% fewer parameters than YOLOv8
- Best CPU performance (important for webcams)
- Superior small object detection (distant skiers)
- Drop-in replacement for YOLOv3

**Implementation:**
```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # Nano for speed
# or
model = YOLO('yolo11m.pt')  # Medium for accuracy
```

#### Secondary Detector: RT-DETR
**Why:**
- Different architecture (transformer vs CNN)
- Good comparison baseline
- Eliminates NMS overhead
- Available in Ultralytics

**Implementation:**
```python
from ultralytics import RTDETR
model = RTDETR('rtdetr-l.pt')
```

#### Optional: YOLO11-Pose
**Why:**
- If you want to track skier movements/posture
- Injury prevention analysis
- Performance metrics

**Implementation:**
```python
from ultralytics import YOLO
model = YOLO('yolo11n-pose.pt')
```

---

## Implementation Priority

### Phase 1: Core Detectors (Immediate)
1. ✅ Keep YOLOv3 (baseline, already implemented)
2. ⭐ Add YOLO11 (primary recommendation)
3. ⭐ Add RT-DETR (transformer alternative)

### Phase 2: Advanced (Optional)
4. Add YOLO11-Pose (if movement tracking needed)
5. Add YOLOv8 (if wider compatibility needed)

### Phase 3: Research (Future)
6. Evaluate RF-DETR (requires custom framework)
7. Fine-tune on ski resort dataset (if available)

---

## Technical Integration Notes

### Ultralytics Models (Easy Integration)
All of these work with your existing architecture:
- YOLO11: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
- RT-DETR: `rtdetr-l.pt`, `rtdetr-x.pt`
- YOLO11-Pose: `yolo11n-pose.pt`, etc.

### Installation
```bash
pip install ultralytics>=8.3.0  # Latest version with YOLO11
```

### Model Download
Models auto-download on first use:
```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # Auto-downloads
```

---

## Conclusion

**For skier detection in webcam counter application:**

✅ **Implement YOLO11** as primary detector
- Best performance/speed trade-off
- Excellent for detecting people at various distances
- Latest technology (Oct 2024)
- Easy integration (Ultralytics)

✅ **Implement RT-DETR** as alternative
- Different architecture for comparison
- Good for complex multi-object scenes
- Transformer-based approach

⚠️ **Skip RF-DETR for now**
- Not integrated in Ultralytics
- Requires separate framework
- Can evaluate later if needed

🎯 **Result:** Your modular architecture makes it trivial to swap between these models!

---

## References

1. "YOLO Model Comparison: YOLOv11 vs Previous" - Ultralytics Blog
2. "DETRs Beat YOLOs on Real-time Object Detection" - Baidu (2023)
3. "Tracking Skiers from the Top to the Bottom" - WACV 2024
4. "SkiTB Visual Tracking Challenge 2025" - arXiv
5. "RF-DETR: A SOTA Real-Time Object Detection Model" - Roboflow Blog
6. "Best Object Detection Models 2025" - Roboflow
7. "YOLOv8 Pose Estimation: Advanced KeyPoint Technology" - Ikomia
8. "EE-YOLOv8: Enhanced Pose Estimation" - Scientific Reports (2025)
