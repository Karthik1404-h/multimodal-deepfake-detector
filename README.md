# 🛡️ Multimodal Deepfake Detector

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA%2012.1-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Target%20GPU-RTX%204050%20(6GB)-76B900.svg)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, **3-way Multimodal Deepfake Detection System** engineered to detect synthetic facial manipulation, temporal sequence anomalies, and AI voice cloning in digital video containers.

Designed for edge deployment on commodity consumer GPUs (e.g., **NVIDIA RTX 4050 6 GB VRAM**), featuring FP16 mixed precision inference, active VRAM management, interactive Streamlit analytics, and calibrated robustness against H.264 video compression artifacts.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
  - [1. Spatial Forensic Branch (EfficientNet-B0)](#1-spatial-forensic-branch-efficientnet-b0)
  - [2. Temporal Anomaly Branch (Bi-LSTM + Attention)](#2-temporal-anomaly-branch-bidirectional-lstm--attention)
  - [3. Audio Forensic Branch (4-Stage Mel-CNN)](#3-audio-forensic-branch-4-stage-mel-cnn)
  - [4. Late-Fusion & Decision Logic](#4-late-fusion--decision-logic)
- [Compression Robustness & Calibration](#-compression-robustness--calibration)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Dataset Acquisition](#-dataset-acquisition)
- [Usage & Execution](#-usage--execution)
  - [Streamlit Web Application](#1-streamlit-interactive-web-dashboard)
  - [CLI Terminal Interface](#2-command-line-interface-cli)
  - [Python API](#3-python-api-integration)
- [Training & Pipeline Workflows](#-training--pipeline-workflows)
- [Hardware & Memory Optimization](#-hardware--memory-optimization)
- [License](#-license)

---

## 🔍 Overview

Modern generative AI tools (FaceSwap, DeepFaceLab, LivePortrait, ElevenLabs, SV2TTS) produce hyper-realistic video and audio forgeries. Single-frame image detectors often fail in production because:
1. **Temporal Discontinuity:** High-quality single frames can still contain temporal jitter, unnatural blinking, and frame-to-frame boundary flickering across sequential frames.
2. **Audio-Visual Disconnect:** Cloned synthetic speech carries spectral anomalies, vocoder artifacts, and unnatural pitch transitions undetectable by visual models alone.
3. **Codec Compression Shift:** Standard H.264/H.265 compression causes macroblocking, quantization noise, and edge softening that cause naive deepfake classifiers to hallucinate high false-positive rates on authentic videos.

**Multimodal Deepfake Detector** solves these challenges through a unified tri-modal pipeline that simultaneously inspects spatial pixels, temporal frame sequences, and audio Mel-spectrograms, reinforced with aggressive training augmentations and post-processing calibration bias.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Input Video Container .mp4 / .avi / .mov] --> B[Video Stream]
    A --> C[Audio Stream]

    subgraph Video Analysis Pipeline
        B --> D[Uniform Frame Sampler]
        D --> E[MediaPipe Face Crop Extractor 224x224 RGB]
        E --> F[EfficientNet-B0 Feature Extractor]
        F --> G[1280-dim Frame Embeddings]
        F --> H[Spatial Classification Head]
        H --> I[Hybrid Spatial Score: 60% Mean + 40% Top-20%]
        G --> J[Bidirectional LSTM with Attention]
        J --> K[Temporal Anomaly Score & Attention Weights]
        I --> L[Video Late-Fusion: 30% Spatial + 70% Temporal]
        K --> L
    end

    subgraph Audio Analysis Pipeline
        C --> M{Audio Track & RMS >= 0.015?}
        M -- Yes (Active Voice) --> N[16kHz Mel-Spectrogram 1x128x128]
        N --> O[4-Stage AudioForensicCNN]
        O --> P[Audio Synthetic Score]
        M -- No (Silent / Noise) --> Q[Bypass Audio Branch]
    end

    subgraph Multimodal Fusion & Verdict
        L --> R{Container Fusion Engine}
        P --> R
        Q --> R
        R --> S[Tri-Modal Max Fusion: max(Video, Audio)]
        R --> T[Bi-Modal Fusion: 100% Video if Silent / Noise Gated]
        S --> U[Calibrated Score]
        T --> U
        U --> V{Tri-State Verdict}
        V -- "Score >= 0.65" --> W[FAKE]
        V -- "0.45 <= Score < 0.65" --> X[UNCERTAIN]
        V -- "Score < 0.45" --> Y[REAL]
    end
```

### 1. Spatial Forensic Branch (EfficientNet-B0)
- **Model:** `SpatialForensicClassifier` (`deepfake_core.models.spatial_efficientnet`)
- **Input:** 224×224 RGB face crops extracted via Google MediaPipe.
- **Backbone:** ImageNet-pretrained `EfficientNet-B0` with frozen feature extractor and trainable binary classification head (Dropout 0.4 → Linear 1280 → 1).
- **Embeddings:** Exposes 1280-dimensional global average pooled feature vectors prior to classification for downstream temporal modeling.
- **Hybrid Scoring:** Calculates a weighted composite:
  $$\text{Spatial Score} = 0.60 \times \text{Mean Frame Prob} + 0.40 \times \text{Top-20\% Worst Frame Prob}$$
  This catches brief fleeting anomalies (e.g., glitch on single frame) while rewarding consistent authenticity.

### 2. Temporal Anomaly Branch (Bidirectional LSTM + Attention)
- **Model:** `TemporalAnomalyLSTM` (`deepfake_core.models.temporal_lstm`)
- **Input:** Sequential frame embeddings $(B, 30, 1280)$ extracted by the spatial backbone.
- **Architecture:** 2-layer Bidirectional LSTM (`hidden_dim=512`, `dropout=0.3`) capturing bidirectional context across time.
- **Attention Mechanism:** A learned sequence attention module computes normalized softmax weights across all time steps, generating an anomaly context vector:
  $$c = \sum_{t=1}^{T} \alpha_t h_t$$
  This pinpoints the precise frames where micro-jitter, lighting shifts, or blinking irregularities occur.
- **Output:** Scalar temporal anomaly score $[0.0, 1.0]$ and per-frame attention distribution.

### 3. Audio Forensic Branch (4-Stage Mel-CNN)
- **Model:** `AudioForensicCNN` (`deepfake_core.models.audio_cnn`)
- **Preprocessing:** `AudioExtractor` (`deepfake_core.preprocessing.audio_extractor`) extracts the audio waveform using `librosa`/`soundfile` at 16 kHz, generates a normalized Mel-spectrogram (`n_mels=128`, `n_fft=1024`, `hop_length=512`), and pads/crops to a uniform $(1, 128, 128)$ tensor.
- **Architecture:** 4 convolutional blocks (Conv2d 3×3, BatchNorm, LeakyReLU, MaxPool2d 2×2, Dropout2d) scaling channels from 32 → 64 → 128 → 256, followed by Global Average Pooling and a classification head.
- **Output:** Synthetic voice probability $[0.0, 1.0]$. Gracefully skipped if the video lacks an audio track.

### 4. Late-Fusion & Decision Logic
The unified inference engine (`UnifiedDetector` in `deepfake_core.pipeline`) implements a resilient multi-stage late fusion:
1. **Video Modality Fusion:**
   $$\text{Score}_{\text{video}} = 0.30 \times \text{Score}_{\text{spatial}} + 0.70 \times \text{Score}_{\text{temporal}}$$
2. **Container Late Fusion (Maximum-Confidence Strategy with Voice Activity Gating):**
   - **Active Voice Audio ($\text{RMS} \ge 0.015$):** $\text{Score}_{\text{final}} = \max(\text{Score}_{\text{video}}, \text{Score}_{\text{audio}})$ (Avoids diluting partial deepfakes where only visual or audio stream is manipulated)
   - **Silent / Ambient Noise ($\text{RMS} < 0.015$) / No Audio:** $\text{Score}_{\text{final}} = \text{Score}_{\text{video}}$ (Prevents ambient room noise from raising false alarms)
3. **Tri-State Calibrated Verdict:**
   - **`FAKE`:** $\text{Score}_{\text{final}} \ge 0.65$ (Stricter threshold preventing false alarms).
   - **`UNCERTAIN`:** $0.45 \le \text{Score}_{\text{final}} < 0.65$ (Requires human-in-the-loop review).
   - **`REAL`:** $\text{Score}_{\text{final}} < 0.45$.

---

## ⚡ Compression Robustness & Calibration

### The Problem: H.264 Compression False Positives
Standard video compression algorithms (H.264/MPEG-4 AVC) employ lossy discrete cosine transforms (DCT), macroblock motion compensation, and 4:2:0 chroma subsampling. Uncalibrated deepfake classifiers mistake these codec compression artifacts for synthetic manipulation, leading to high false-positive rates on authentic videos.

### The Solution: Two-Pronged Robustness Defense
1. **Video Degradation Training Augmentations:**
   In [`scripts/train_spatial.py`](file:///d:/Projects/Multimodal%20Deepfake%20Detector/scripts/train_spatial.py), the training pipeline applies aggressive torchvision transforms simulating compression:
   - `ColorJitter` (brightness, contrast, saturation, hue variance)
   - `RandomAutocontrast` (simulating codec quantization dynamics)
   - `GaussianBlur` (simulating bitrate reduction and motion blur)
   - `RandomAdjustSharpness` (both softening at 0.6× and edge ringing at 1.8×)
   - `AddGaussianNoise` (zero-mean sensor and high-frequency noise injection)
   - *Applied strictly to training; validation transforms remain unaugmented.*
2. **Live Inference Calibration Bias:**
   In [`deepfake_core/pipeline.py`](file:///d:/Projects/Multimodal%20Deepfake%20Detector/deepfake_core/pipeline.py), `UnifiedDetector` applies a `calibration_bias` parameter (default `0.10`):
   $$\text{Prob}_{\text{calibrated}} = \text{clip}(\text{Prob}_{\text{raw}} - \text{calibration\_bias}, 0.0, 1.0)$$
   This offsets the compression domain shift, pushing borderline compressed frames into authentic probability space.

---

## 📁 Project Structure

```
Multimodal Deepfake Detector/
├── deepfake_core/                     # Core installable Python package
│   ├── __init__.py                    # Public API (UnifiedDetector, models)
│   ├── __main__.py                    # Module entry point
│   ├── cli.py                         # Command-line interface
│   ├── pipeline.py                    # Unified multimodal detection pipeline
│   ├── data/                          # PyTorch Dataset loaders
│   │   ├── __init__.py
│   │   ├── audio_dataset.py           # DeepfakeAudioDataset (Mel-spectrograms)
│   │   └── temporal_dataset.py        # TemporalEmbeddingDataset (PT sequences)
│   ├── models/                        # Neural network architectures
│   │   ├── __init__.py
│   │   ├── audio_cnn.py               # AudioForensicCNN (4-stage Conv2D)
│   │   ├── spatial_efficientnet.py    # SpatialForensicClassifier (EfficientNet-B0)
│   │   └── temporal_lstm.py           # TemporalAnomalyLSTM (Bi-LSTM + Attention)
│   ├── preprocessing/                 # Extraction and transformation utilities
│   │   ├── __init__.py
│   │   ├── audio_extractor.py         # Audio extraction & Mel-spectrogram generator
│   │   └── face_extractor.py          # MediaPipe face detection & cropping
│   └── utils/
│       ├── __init__.py
│       └── vram_manager.py            # CUDA memory monitoring & garbage collection
│
├── models/
│   └── checkpoints/                   # Trained neural network weights (.pth)
│       ├── best_spatial_model.pth     # Spatial classifier weights (~16 MB)
│       ├── best_temporal_model.pth    # Temporal LSTM weights (~55 MB)
│       └── best_audio_model.pth       # Audio Mel-CNN weights (~1.6 MB)
│
├── scripts/                           # Training, extraction, and utility scripts
│   ├── download_from_drive.py         # Google Drive dataset sync utility
│   ├── extract_embeddings.py          # Frame embedding pre-computation script
│   ├── generate_report.py             # Evaluation report generator
│   ├── ingest_huggingface_data.py     # Hugging Face dataset streamer
│   ├── predict_video.py               # Standalone video inference script
│   ├── preprocess_dataset.py          # Batch face crop extraction script
│   ├── process_local_videos.py        # Local folder video processor
│   ├── train_audio.py                 # Training script for AudioForensicCNN
│   ├── train_spatial.py               # Training script for SpatialForensicClassifier
│   ├── train_temporal.py              # Training script for TemporalAnomalyLSTM
│   └── verify_fusion.py               # Benchmark verification & diagnostic script
│
├── data/                              # Datasets and cache storage (gitignored)
│   ├── processed/                     # Face crops and extracted embeddings
│   └── raw/                           # Raw videos and audio files
│
├── app.py                             # Interactive Streamlit Web Application
├── download_dfdc_sample.py            # Kaggle dataset fetcher (kagglehub)
├── pyproject.toml                     # Python package metadata & dependencies
├── requirements.txt                   # Dependency manifest
└── README.md                          # Documentation (this file)
```

---

## 💻 Installation & Setup

### 1. Prerequisites
- **Operating System:** Windows 10/11 or Ubuntu 20.04+
- **Python:** 3.10 or 3.11
- **CUDA:** 11.8 or 12.1+ (with NVIDIA driver 530+)

### 2. Virtual Environment Setup
Open a terminal in the project directory:

```bash
# Clone the repository
git clone https://github.com/Karthik1404-h/multimodal-deepfake-detector.git
cd "multimodal-deepfake-detector"

# Create a clean virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (cmd.exe):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA Support
Install PyTorch matching your CUDA hardware (recommended: CUDA 12.1 for RTX 40-series GPUs):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies & `deepfake_core`
Install all required libraries and install `deepfake_core` in editable development mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Verify your setup:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 📦 Dataset Acquisition

The project is preconfigured to stream and train on benchmark datasets:

### Option A: Kaggle Dataset Fetcher (Automated)
Run the built-in downloader to fetch the `manjilkarki/deepfake-and-real-images` dataset via `kagglehub`:

```bash
python download_dfdc_sample.py
```
This authenticates with Kaggle, downloads the face crop dataset (< 1 GB), and organizes it directly into `data/processed/train/real/` and `data/processed/train/fake/`.

### Option B: Audio Deepfake Dataset
Place audio benchmarks (e.g., FakeAVCeleb or WaveFake) into `data/raw/audio_deepfake/`:
```
data/raw/audio_deepfake/
├── original/       # Authentic audio recordings (.wav)
└── synthetic/      # AI-synthesized or voice-cloned audio (.wav)
```

---

## 🚀 Usage & Execution

### 1. Streamlit Interactive Web Dashboard

Launch the web application (ensure your virtual environment is active):

```bash
# Windows (PowerShell with venv active):
streamlit run app.py

# Or directly using the venv binary:
.\.venv\Scripts\streamlit.exe run app.py
```

The application opens in your default browser at `http://localhost:8501`.

#### Key Dashboard Capabilities:
- 📤 **Video Upload:** Upload any `.mp4`, `.avi`, or `.mov` video or select from pre-loaded authentic and synthetic samples.
- 🎯 **Verdict Card:** Live badge displaying **FAKE**, **UNCERTAIN**, or **REAL** with overall confidence percentage.
- 📊 **Modality Breakdown:** Side-by-side metric cards for Spatial Score, Temporal Score, and Audio Forensic Score (with ambient noise gating indicators).
- 📈 **Temporal Attention Anomaly Map:** Interactive Plotly graph showing frame-by-frame attention weights and identifying the exact timestamp of peak manipulation.
- 🎚️ **Interactive Tuning:** Sidebar sliders for Fake Threshold, Uncertain Threshold, Frame Sampling Budget, Voice Activity Audio RMS Gate, and **Compression Calibration Bias**.
- 📋 **Forensic Telemetry:** Expandable panel with raw JSON payload, mean frame probability, and top-20% frame metrics.

---

### 2. Benchmark Verification Script

Run the automated multimodal diagnostic check on benchmark real and deepfake video clips:

```bash
python scripts/verify_fusion.py
```

---

### 3. Command-Line Interface (CLI)

Run inference directly from PowerShell or Bash:

```bash
# Run on GPU with default calibrated settings
deepfake-detect --video "path/to/suspect_video.mp4" --device cuda

# Or run via Python module
python -m deepfake_core.cli --video "data/raw/deepfake_videos/real_test_video.mp4" --device cuda

# Run with custom calibration bias and threshold
python -m deepfake_core.cli \
    --video "suspect.mp4" \
    --device cuda \
    --threshold 0.70 \
    --calibration-bias 0.15 \
    --max-frames 40
```

#### CLI Output Example:
```json
{
  "video": "real_test_video.mp4",
  "verdict": "REAL",
  "is_fake": false,
  "final_score": 0.4124,
  "confidence": 0.5876,
  "calibration_bias": 0.10,
  "video_score": 0.4350,
  "spatial_score": 0.4700,
  "temporal_score": 0.4000,
  "audio_score": 0.3600,
  "has_audio": true,
  "frames_analyzed": 30,
  "device": "cuda"
}
```

---

### 3. Python API Integration

Integrate the detector directly into your Python services or surveillance pipelines:

```python
from deepfake_core import UnifiedDetector

# Initialize detector on CUDA (auto-loads best checkpoints)
detector = UnifiedDetector(
    device="cuda",
    threshold=0.65,
    uncertain_threshold=0.45,
    calibration_bias=0.10,
    max_frames=30
)

# Run multimodal analysis
result = detector.predict_video_clip("path/to/video.mp4")

print(f"Verdict: {result['verdict']} (Confidence: {result['confidence']:.2%})")
print(f"Spatial Score: {result['spatial_score']}")
print(f"Temporal Score: {result['temporal_score']}")
print(f"Audio Score: {result['audio_score']}")
```

---

## 🏋️ Training & Pipeline Workflows

To retrain the complete multimodal pipeline from scratch:

### Step 1: Train Spatial Forensic Classifier
Train `EfficientNet-B0` with aggressive compression augmentations:
```bash
python scripts/train_spatial.py \
    --data-dir data/processed/train \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda
```
*Best checkpoint saves to `models/checkpoints/best_spatial_model.pth`.*

### Step 2: Pre-compute Temporal Embeddings
Extract 1280-dim sequence feature representations to speed up temporal training:
```bash
python scripts/extract_embeddings.py \
    --data-dir data/processed/train \
    --output-dir data/processed/embeddings/train \
    --batch-size 32 \
    --device cuda
```

### Step 3: Train Temporal Anomaly Bi-LSTM
Train the sequential LSTM with attention on the extracted embeddings:
```bash
python scripts/train_temporal.py \
    --data-dir data/processed/embeddings/train \
    --epochs 25 \
    --batch-size 64 \
    --lr 5e-4 \
    --device cuda
```
*Best checkpoint saves to `models/checkpoints/best_temporal_model.pth`.*

### Step 4: Train Audio Forensic Mel-CNN
Train the 4-stage convolutional network on Mel-spectrograms:
```bash
python scripts/train_audio.py \
    --data-dir data/raw/audio_deepfake \
    --epochs 20 \
    --batch-size 32 \
    --lr 3e-4 \
    --device cuda
```
*Best checkpoint saves to `models/checkpoints/best_audio_model.pth`.*

---

## ⚙️ Hardware & Memory Optimization

Engineered specifically for budget-conscious edge devices (e.g., **NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM**):

| Optimization Strategy | Implementation Details |
|---|---|
| **FP16 Mixed Precision** | `torch.amp.autocast('cuda')` used across all inference forward passes, cutting memory bandwidth by 50%. |
| **Active VRAM Flushing** | Explicit invocation of `torch.cuda.empty_cache()` and `gc.collect()` at the conclusion of every video inference cycle. |
| **Embedding Caching** | Separating spatial feature extraction from temporal sequence modeling eliminates backpropagation through EfficientNet during temporal training. |
| **Efficient Footprint** | EfficientNet-B0 (~4M params), Temporal LSTM (~13.7M params), and Audio CNN (~400k params) fit easily within < 2.5 GB peak VRAM during live inference. |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
