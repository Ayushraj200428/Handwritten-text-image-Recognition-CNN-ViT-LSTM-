# Hindi Handwritten Text Recognition (HTR)

> A data-efficient hybrid deep learning system for real-time Hindi handwritten text recognition, combining CNN, Vision Transformer (ViT), and BiLSTM architectures with a CTC decoder.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Model Comparison](#model-comparison)
- [Pretrained Model](#pretrained-model)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [API Reference](#api-reference)
- [Dataset](#dataset)
- [Team](#team)

---

## Overview

This project implements a production-ready OCR system capable of recognizing handwritten Devanagari script. The system is served via a Flask web application with a real-time upload and inference UI.

**Key highlights:**
- 109-class Devanagari character set including vowels, consonants, matras, and numerals
- Hybrid CNN + ViT + BiLSTM architecture trained end-to-end with CTC loss
- 96.4% confidence accuracy on the test set after 38 epochs
- ~50ms inference on CPU, sub-20ms on GPU
- Supports image upload and live webcam capture

---

## Architecture

```
Input Image (1 × 64 × 256)
        │
        ▼
┌─────────────────────┐
│  CNN Feature         │  6 conv layers, 2× MaxPool → (256 × 16 × 64)
│  Extractor           │  Extracts local stroke and edge features
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Token Embedding     │  Conv2d patch projection (patch_size=4)
│  + Pos Encoding      │  → (B × 64 tokens × 384 dims)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ViT Encoder         │  6 Transformer blocks, 6 heads
│                      │  Captures global character co-occurrence context
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  BiLSTM Sequence     │  2-layer BiLSTM (hidden=256, bidirectional)
│  Modeler             │  Adds left-right sequential reading order
│                      │  Residual connection back to ViT output
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  CTC Prediction Head │  Linear → GELU → Dropout → Linear → 109 classes
└─────────────────────┘
        │
        ▼
   CTC Decode → Text string + Confidence score
```

**Why BiLSTM after ViT?**
ViT captures global context (which characters appear together), while BiLSTM captures local reading order (left-to-right stroke sequence). Devanagari matras depend heavily on adjacent characters — BiLSTM handles this naturally. A residual connection ensures ViT features are not lost.

---

## Model Comparison

| Model | Accuracy | Inference | Params | Notes |
|---|---|---|---|---|
| CNN only | ~83% | ~44ms | 5M | No sequence context |
| Vision Transformer | ~91% | ~40ms | 13.8M | Global attention, data hungry |
| Hybrid CNN + ViT | ~93% | ~40ms | 13.8M | No sequential modeling |
| **Hybrid CNN + ViT + LSTM** | **96.4%** | **~50ms** | **16.9M** | **Best model** |

---

## Pretrained Model

The trained `LSTM_VERSION.keras` checkpoint is hosted on Kaggle:

**[Download — CNN + ViT + LSTM (Kaggle)](https://www.kaggle.com/models/ayushbuchu/cnn-vit-lstm-model)**

After downloading, place the file in the project root:

```
hindi-htr/
└── LSTM_VERSION.keras   ← place here
```

Then run the app normally:

```bash
python app.py
```

---

## Project Structure

```
├── app.py                      # Flask web server, inference endpoints
├── Major_Project.py            # Model architecture, training, dataset
├── Major_Project.ipynb         # Jupyter notebook (training + experiments)
├── LSTM_VERSION.keras          # Trained model checkpoint (best)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container setup
├── templates/
│   └── index.html              # Frontend UI
├── static/
│   ├── app.js                  # Frontend logic
│   └── styles.css              # Styling
└── archive/
    └── labels.csv              # Character label mapping
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/hindi-htr.git
cd hindi-htr

# Install dependencies
pip install -r requirements.txt
```

### Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

### Run with Docker

```bash
docker build -t hindi-htr .
docker run -p 7860:7860 hindi-htr
```

---

## Training

Training requires two Parquet files:

| File | Description |
|---|---|
| `sikhna.parquet` | Training set (Hindi handwriting images + labels) |
| `pariksha.parquet` | Test / validation set |

```bash
python Major_Project.py
```

You will be prompted to choose a mode:

```
1. Train from scratch
2. Load model + run webcam inference
3. Train then run inference
```

**Training config (auto-selected based on device):**

| Setting | GPU | CPU |
|---|---|---|
| Batch size | 32 | 16 |
| Epochs | 50 | 20 |
| Patch size | 4 | 8 |
| Max samples | all | 5,000 train / 1,000 test |
| Early stopping | patience=15, min_delta=1e-4 | same |

The best checkpoint is saved to `LSTM_VERSION.keras`.

---

## API Reference

### `GET /health`

Returns model load status.

```json
{
  "ok": true,
  "service": "hindi-ocr-minimal",
  "device": "cpu",
  "model_loaded": true,
  "model_error": null
}
```

### `POST /predict`

Run OCR on a base64-encoded image.

**Request body:**
```json
{
  "file": "<base64 encoded image or data URL>"
}
```

**Response:**
```json
{
  "success": true,
  "text": "अभिलेख",
  "confidence": 0.964,
  "timestamp": "18:30:00",
  "status": "ok"
}
```

**Status values:** `ok` · `low_confidence` · `no_text_region`

---

## Dataset

The dataset is hosted on Kaggle:

**[Download — Handwritten Text Recognition Dataset (Kaggle)](https://www.kaggle.com/datasets/ayushbuchu/hand-written-text-recognition)**

After downloading, place the files in the project root:

```
hindi-htr/
├── sikhna.parquet     ← training set
└── pariksha.parquet   ← test set
```

The model is trained on a Parquet-formatted Hindi handwriting dataset.

- **Image size:** 64 × 256 (H × W), grayscale
- **Character set:** 109 Devanagari classes (vowels, consonants, matras, numerals, punctuation + `<BLANK>`)
- **Preprocessing:** CLAHE → aspect-ratio resize → white padding → normalize to [0, 1]

**Augmentations used during training:**
- Random rotation ±10°
- Horizontal shear
- Elastic distortion
- Morphological dilation / erosion (pen thickness simulation)
- Gaussian + median blur
- Brightness & contrast jitter
- Gaussian noise
- Random perspective warp

---

## Team

| Name | PRN |
|---|---|
| Saurabh Pinjarkar | 202301060013 |
| Harshal Devkate | 202301060016 |
| Ayush Raj | 202301100066 |
| Rishit Ujjain | 202301040280 |

---

## Tech Stack

- **Model:** PyTorch, timm
- **Server:** Flask
- **CV:** OpenCV, Pillow
- **Training:** AMP (mixed precision), AdamW, ReduceLROnPlateau, CTC Loss
- **Container:** Docker
