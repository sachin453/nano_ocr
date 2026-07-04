# UltraOCR

A lightweight, config-driven OCR system built with PyTorch. UltraOCR combines a **truncated timm pretrained backbone** (default: MobileNetV3-Small) with **multi-scale feature fusion**, a **BiGRU** sequence model, and a **CTC** decoder to recognize text from cropped word images. It ships with a synthetic data generator, real-dataset ingestion helpers, and simple train/eval scripts.

## Key Features

- **Lightweight backbone** — keeps only the first N InvertedResidual blocks of a timm model for a small footprint.
- **Multi-scale fusion** — optionally fuses features from multiple backbone blocks via 1×1 convs + upsampling.
- **Config-driven** — model, charset, dataset, and training are all controlled via a single YAML file.
- **Synthetic data generation** — multiprocess generator with rich augmentations (noise, blur, perspective, JPEG artifacts, elastic distortion, and more).
- **Real dataset support** — helpers to ingest ICDAR 2015 and IIIT5K.
- **CTC training** — AdamW + cosine annealing LR, early stopping, best/latest checkpointing.
- **Character accuracy eval** — Levenshtein-distance-based accuracy metric.

## Architecture

```
Input (N, 3, 32, 256)
        │
        ▼
Truncated timm backbone (MobileNetV3-Small, first 4 blocks)
        │   (optional) multi-scale fusion → (N, C, H, W)
        ▼
Mean-pool height → (N, C, W)
        │
        ▼
Transpose → (N, T, C)
        │
        ▼
BiGRU → (N, T, 2*hidden)
        │
        ▼
Linear decoder → (N, T, num_chars + 1)   # CTC logits
```

The backbone outputs a feature map which is collapsed along the height dimension (mean pool), transposed to a sequence, passed through a bidirectional GRU, and projected to character logits. CTC loss with greedy decoding is used for training and inference.

## Project Structure

```
ultraocr/
├── config/
│   └── ocr_mobilenetv3.yaml      # Default model/dataset/training config
├── scripts/
│   ├── generate_synthetic.py     # Multiprocess synthetic data generator
│   ├── build_datasets.py         # Ingest ICDAR 2015 / IIIT5K
│   ├── train.py                  # Training entry point
│   └── eval.py                   # Evaluation entry point
├── ultraocr/
│   ├── __init__.py
│   ├── config.py                 # YAML config loader
│   ├── model.py                  # OCR model (backbone + BiGRU + CTC decoder)
│   ├── dataset.py                # Disk-backed OCR dataset
│   ├── utils.py                  # CTC decode, Levenshtein, collate, colors
│   └── backbones/
│       └── __init__.py           # Truncated timm backbone + multi-scale fusion
├── artifacts/
│   ├── aocr_best.pt              # Best checkpoint
│   └── aocr_latest.pt            # Latest checkpoint
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/sachin453/nano_ocr.git
cd nano_ocr
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Fonts (for synthetic generation)

The synthetic generator auto-discovers `.ttf`/`.otf` fonts from these directories:

- `data/fonts_simple/`
- `data/fonts_heavy/`
- `data/fonts/`

Place font files in any of these before generating data.

## Configuration

All settings live in `config/ocr_mobilenetv3.yaml`. Key sections:

```yaml
model:
  timm_name: "mobilenetv3_small_100.lamb_in1k"   # any timm model name
  num_blocks: 4                                   # keep first N InvertedResidual blocks
  pretrained: true
  gru_hidden_size: 128
  gru_num_layers: 1
  multi_scale: true
  fusion_channels: 48
  img_h: 32
  img_w: 256

charset: "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?:;'\"()[]{}+-=/%*$€£¥₹@#&_|/\\<>~^"
blank_token: 0

dataset:
  json_path: "data/synthetic/labels.json"
  num_samples: 200000
  train_split: 0.85
  batch_size: 512
  num_workers: 10
  synthetic:
    text_length: [2, 15]
    font_size: [20, 48]
    pad: [2, 8]
    jitter: [0, 15]

training:
  lr: 0.001
  weight_decay: 0.00001
  epochs: 50
  early_stopping_patience: 50

checkpoint:
  dir: "artifacts"
  latest_name: "aocr_latest.pt"
  best_name: "aocr_best.pt"
```

To use a different backbone or charset, copy this file and pass `--config <name>` (without the `.yaml` extension) to the scripts.

## Usage

### 1. Generate Synthetic Data

Creates a pre-built dataset of augmented word images and a `labels.json` index.

```bash
python scripts/generate_synthetic.py \
    --config ocr_mobilenetv3 \
    --count 200000 \
    --output data/synthetic \
    --workers 10
```

Options:
- `--count` — number of images to generate (default: 1,000,000)
- `--output` — output directory (default: `data/synthetic`)
- `--workers` — parallel worker processes (default: CPU count)

### 2. Train

```bash
python scripts/train.py --config ocr_mobilenetv3
```

Trains on the dataset referenced by `dataset.json_path`, splits into train/val by `train_split`, uses AdamW + cosine annealing, and saves checkpoints to `artifacts/`. Early stopping is controlled by `training.early_stopping_patience`.

### 3. Evaluate

```bash
python scripts/eval.py --config ocr_mobilenetv3
```

Loads the best checkpoint (`artifacts/aocr_best.pt` by default) and reports character accuracy on the dataset. Use `--checkpoint <path>` to evaluate a specific checkpoint and `--num-samples <n>` to print extra synthetic-sample predictions.

### 4. Build Real Datasets

`scripts/build_datasets.py` ingests ICDAR 2015 and IIIT5K into a unified `train_labels.json`. Edit the folder paths at the bottom of the script first:

```python
icdar_folder = "./data/icdar2015/"
iiit5k_folder = "./data/iiit5kwords/"
icdar_crops_folder = "./data/icdar2015_crops/"
```

Then run:

```bash
python scripts/build_datasets.py
```

## Checkpoints

- `artifacts/aocr_best.pt` — best model by character accuracy
- `artifacts/aocr_latest.pt` — latest model after each epoch

## Dependencies

- torch
- torchvision
- timm
- torchinfo
- numpy
- opencv-python
- pillow
- scipy
- tqdm
- pyyaml

See `requirements.txt` for the full list.