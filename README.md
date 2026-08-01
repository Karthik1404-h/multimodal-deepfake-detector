# deepfake_core

**Modular multimodal deepfake detection library** — spatial, temporal, and lipsync forensic analysis, optimized for edge deployment on NVIDIA RTX hardware.

> **Version:** 0.1.0  
> **Python:** 3.10+  
> **Hardware Target:** NVIDIA RTX 4050 (6 GB VRAM), i7 13th Gen, NVMe SSD

---

## Features

- 🔬 **Spatial Forensic Analysis** — EfficientNet-B0 backbone for frame-level real/fake classification
- 👤 **Face Extraction Pipeline** — MediaPipe-powered face detection with configurable padding and crop size
- ⚡ **VRAM-Optimized** — FP16 mixed precision, explicit GPU cache management for 6 GB cards
- 📦 **Pip-Installable** — Clean `pip install -e .` for integration into downstream projects (e.g., CCTV Surveillance)
- 🖥️ **CLI Interface** — Run inference directly from the terminal

---

## Quick Start

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd "Multimodal Deepfake Detector"

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate
```

### 2. Install PyTorch with CUDA

```bash
# For CUDA 12.1 (RTX 4050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install `deepfake_core` (Editable Mode)

```bash
pip install -e .
```

This installs the package in development mode — any code changes take effect immediately without reinstalling.

### 4. Verify Installation

```bash
python -c "from deepfake_core import UnifiedDetector; print('✅ deepfake_core installed successfully')"
```

---

## CLI Usage

```bash
# Run inference on a video
deepfake-detect --video path/to/video.mp4 --device cuda

# Or via module
python -m deepfake_core.cli --video path/to/video.mp4 --device cuda

# With custom threshold
deepfake-detect --video path/to/video.mp4 --device cuda --threshold 0.7

# Use a custom model checkpoint
deepfake-detect --video path/to/video.mp4 --model-path path/to/weights.pt
```

### Output Example

```json
{
  "video": "sample.mp4",
  "is_fake": true,
  "confidence": 0.934,
  "spatial_score": 0.934,
  "lipsync_score": 0.0,
  "device": "cuda",
  "frames_analyzed": 30
}
```

---

## Python API

```python
from deepfake_core import UnifiedDetector

detector = UnifiedDetector(device="cuda")
result = detector.predict_video_clip("sample.mp4")

print(result)
# {
#     "is_fake": True,
#     "confidence": 0.934,
#     "spatial_score": 0.934,
#     "lipsync_score": 0.0
# }
```

---

## Project Structure

```
Multimodal Deepfake Detector/
├── deepfake_core/              # Installable Python package
│   ├── __init__.py             # Public API exports
│   ├── pipeline.py             # UnifiedDetector — main entry point
│   ├── cli.py                  # CLI interface (argparse)
│   ├── models/
│   │   └── spatial_efficientnet.py   # EfficientNet-B0 binary classifier
│   ├── preprocessing/
│   │   └── face_extractor.py         # MediaPipe face crop extraction
│   └── utils/
│       └── vram_manager.py           # GPU memory utilities
├── data_cache/                 # Hot SSD storage (gitignored)
│   ├── processed_faces/        # 224×224 face crops
│   └── sample_videos/          # Test video clips
├── scripts/
│   ├── download_from_drive.py  # Google Drive ↔ local sync
│   ├── preprocess_dataset.py   # Batch face extraction
│   └── train_spatial.py        # Training loop
├── tests/                      # Unit tests
├── pyproject.toml              # Package build config
├── requirements.txt            # Dependency listing
└── README.md                   # This file
```

---

## Data Architecture: Cold Vault vs. Hot SSD

This project uses a two-tier storage strategy to manage large video datasets efficiently:

### 🧊 Cold Vault — Google Drive

- **Purpose:** Long-term archival of raw datasets (FaceForensics++, Celeb-DF, DFDC, etc.)
- **Contents:** Original MP4/AVI clips, compressed archives, dataset metadata
- **Access:** Via `scripts/download_from_drive.py` (uses `gdown` for authenticated downloads)
- **Why:** Raw datasets are 100+ GB — storing them on Drive keeps git clean and enables team sharing

### 🔥 Hot SSD — `data_cache/`

- **Purpose:** Fast local storage for preprocessed, training-ready data
- **Contents:**
  - `processed_faces/` — 224×224 RGB face crops (PNG) extracted from raw videos
  - `sample_videos/` — Small test clips for development and CI
- **Access:** Direct filesystem read, optimized for NVMe SSD throughput
- **Why:** Training and inference need low-latency random access — NVMe delivers ~7 GB/s sequential read

### Workflow

```
Google Drive (Cold)  ──download──▶  raw videos  ──preprocess──▶  data_cache/ (Hot SSD)
                                                                      │
                                                              Training / Inference
```

---

## Hardware Optimization Notes

| Feature | Implementation |
|---|---|
| **Mixed Precision (FP16)** | `torch.cuda.amp.autocast` in inference and training |
| **VRAM Management** | Explicit `torch.cuda.empty_cache()` + `gc.collect()` after every inference batch |
| **Batch Size** | Auto-tuned for 6 GB VRAM budget |
| **Model Size** | EfficientNet-B0 ≈ 20 MB weights, ~4M parameters |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check deepfake_core/
```

---

## License

MIT
