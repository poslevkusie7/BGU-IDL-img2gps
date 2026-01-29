# Image-to-GPS Regression

We study image-based localization on the Ben Gurion University campus by predicting geographic coordinates from a single RGB image. We consider the task as a coordinate regression using a ViT DINOv2 backbone pretrained and a lightweight regression head. We also consider a multi-head variant with the same shared encoder, but propose adding another classification head to regularize the representation. We evaluate the performance in meters using mean distance error and the percentage of predictions within 10m and 25m of the ground truth.

## Quick Start 
Run these commands **line by line** after cloning the repo:
```bash
chmod +x setup.sh
./setup.sh
conda activate img2gps
```

Place the dataset and checkpoint:
```bash
mkdir -p data/images
# copy images into data/images/
# copy gt.csv into data/gt.csv
# copy a best.pt into runs/coords_latlon/best.pt
```

Set the checkpoint path:
```bash
export IMG2GPS_CHECKPOINT=runs/coords_latlon/best.pt
```

Smoke test (single image via submission API):
```bash
python - <<'PY'
import numpy as np
from PIL import Image
from submission import predict_gps

img = np.array(Image.open("data/images/example.jpg").convert("RGB"))
print(predict_gps(img))
PY
```

Batch evaluation with metrics:
```bash
python -m src.infer \
  --image-dir data/images \
  --gt-csv data/gt.csv \
  --checkpoint runs/coords_latlon/best.pt
```

## Overview
- Task: regress absolute GPS latitude/longitude from a single RGB image.
- Backbone: DINOv2 ViT (via timm) with a lightweight regression head.
- Variant: multi-head model with an auxiliary classification head for regularization.
- Metrics: mean/median distance error in meters, P@10m, P@25m.

## Repository Layout
- `src/` training, model, and inference code.
- `submission.py` required `predict_gps(image)` API for automatic evaluation.
- `config/` example training config.
- `data/` dataset placeholders, CSVs, and splits.
- `utils/` data prep and visualization helpers.
- `setup.sh` conda-based environment setup.

## Dataset
Per the submission regulations, GPS datasets must follow:
```
dataset_root/
├── images/
└── gt.csv
```

Notes for this repo:
- `data/images/` is the expected image folder.
- `data/gt.csv` contains the required 3 columns: image filename, latitude, longitude.
- `data/metadata.csv` (training CSV) includes an extra `sector_label` column for the multi-head classifier.
- The full image dataset is stored on the shared drive (not in this repo).

## Environment Setup
Python 3.9+ is recommended.

Option A — Conda (recommended for GPU):
```bash
chmod +x setup.sh
./setup.sh
conda activate img2gps
```

Option B — venv + pip:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch install tips:
- If you need CUDA, install the matching PyTorch wheel for your system before `pip install -r requirements.txt`.

## Training
Regression-only (lat/lon):
```bash
python -m src.train \
  --task coords \
  --csv-path data/metadata.csv \
  --img-dir data/images \
  --coord-mode latlon \
  --coord-norm standard \
  --batch-size 16 \
  --pretrained
```

Multi-head (classification + regression) using config:
```bash
python -m src.train --config config/example.yaml
```

Checkpoints are saved based on the config (see `config/example.yaml`).

## Testing / Inference
Batch inference + metrics:
```bash
python -m src.infer \
  --image-dir data/images \
  --gt-csv data/gt.csv \
  --checkpoint runs/coords_latlon/best.pt
```

Single image:
```bash
python -m src.infer \
  --image data/images/your_image.jpg \
  --gt-csv data/gt.csv \
  --checkpoint runs/coords_latlon/best.pt
```

## Submission API 
The evaluation system will call:
```python
def predict_gps(image: np.ndarray) -> np.ndarray:
    ...
```
- Input: `numpy.ndarray` of shape `(H, W, 3)`, RGB, `uint8`, range `[0,255]`
- Output: `numpy.ndarray` of shape `(2,)`, `float32`, `[latitude, longitude]`

Implementation is provided in `submission.py`.

Environment variables supported:
- `IMG2GPS_CHECKPOINT`: path to the checkpoint to load.
- `IMG2GPS_DEVICE`: `cpu`, `cuda`, or `auto` (default `auto`).
- `IMG2GPS_IMG_SIZE`: resize input (default `518`).
