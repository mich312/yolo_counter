# PaddleOCR Setup Guide

## Overview

PaddleOCR is **3-5x faster** than EasyOCR for timestamp extraction, with a much smaller model size (<10MB vs ~100MB).

## Installation

```bash
pip install -r requirements.txt
```

This installs `paddleocr>=2.8.0,<3.0.0` (v2.8-2.10 recommended for better offline support).

## First Run: Model Download

**IMPORTANT:** PaddleOCR requires **internet access on first run** to download models (~8-10MB).

Models are downloaded to `~/.paddleocr/whl/` and cached for offline use.

### What Gets Downloaded

- **Detection model** (~3MB): Text detection
- **Recognition model** (~5MB): Text recognition
- Total: ~8-10MB one-time download

### Download Sources

- Primary: `paddleocr.bj.bcebos.com`
- Fallback: HuggingFace, ModelScope

## Usage

### Option 1: PaddleOCR (default, faster)

```bash
# Run with PaddleOCR (default)
python main.py
```

### Option 2: EasyOCR (legacy, slower)

```bash
# Use EasyOCR instead
export OCR_ENGINE=easy
python main.py
```

## Performance Comparison

| OCR Engine | Speed (timestamp) | Model Size | First Run |
|------------|------------------|------------|-----------|
| **PaddleOCR** ⭐ | **50-100ms** | <10MB | Needs internet |
| EasyOCR | 150-300ms | ~100MB | Needs internet |

## Troubleshooting

### Error: "Downloading from ... failed with code 403"

**Cause:** No internet access or firewall blocking downloads

**Solutions:**
1. Ensure internet connection
2. Check firewall/proxy settings
3. Use EasyOCR as fallback: `export OCR_ENGINE=easy`

### Error: "No module named 'paddle'"

**Cause:** PaddlePaddle not installed

**Solution:**
```bash
pip install paddlepaddle
```

### Offline Usage

After first successful run, models are cached locally. No internet needed for subsequent runs.

## Testing

```bash
# Run PaddleOCR tests (requires internet on first run)
python test_ocr_paddle.py
```

**Test Results (with internet):**
- ✓ Module imports
- ✓ Initialization
- ✓ OCR reader loading
- ✓ Timestamp extraction
- ✓ Config integration

## Configuration

In `.env` file:

```bash
# OCR Engine (paddle or easy)
OCR_ENGINE=paddle

# OCR Languages (comma-separated)
OCR_LANGUAGES=de,en

# Enable/disable OCR
OCR_ENABLED=true
```

## Architecture

```
src/
├── ocr.py           # EasyOCR implementation (legacy)
├── ocr_paddle.py    # PaddleOCR implementation (faster) ⭐
└── config.py        # OCR engine selection logic
```

**Key Features:**
- Drop-in replacement for EasyOCR
- Same interface/API
- 3-5x faster
- Smaller models
- Configurable via environment variables

## Why PaddleOCR?

1. **Faster**: 3-5x speedup for timestamp extraction
2. **Smaller**: <10MB models vs ~100MB for EasyOCR
3. **Production-ready**: Used in production by Baidu
4. **Well-maintained**: Active development, latest v2.10.0
5. **Flexible**: Supports 80+ languages

## Migration from EasyOCR

No code changes needed! PaddleOCR is a drop-in replacement:

```python
# Works with both engines automatically
from src.config import AppConfig
config = AppConfig.from_env()

# OCR engine selected based on OCR_ENGINE env var
# Default: paddle (faster)
```

## Version Notes

- **v2.8-2.10** (recommended): Better offline support, stable API
- **v3.x+**: Newer but requires more network dependencies on first run

The `requirements.txt` specifies `paddleocr>=2.8.0,<3.0.0` for best compatibility.
