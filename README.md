# BGU-IDL-img2gps
Image-to-GPS regression on university campus: training and evaluating models that map images to precise geocoordinates.

## Requirements
- Python 3.9+
- PyTorch, torchvision
- timm (for DINOv2 and Swin backbones)
- pandas, pillow
- utm (only if you use `--coord-mode utm`)

## Data
CSV must include: `image_id`, `sector_label`, `lat`, `lon`.

Expected layout:
```
data/
  metadata1.csv
  images/
    *.jpg
```

## Training
In the root of the project: 
```
python -m src.train --task coords --csv-path data/metadata1.csv --img-dir data/images --pretrained --amp --coord-mode utm --coord-norm standard --batch-size 16

```

## Notes
- `sector_label` is auto-factorized to `0..N-1`.
- `--coord-mode latlon` + `--coord-norm standard` enables statistical normalization for regression.
- DINOv2 uses 518x518 inputs; reduce `--batch-size` if you see CUDA OOM.
- For fewer HF download warnings, set `HF_TOKEN` before the first run.
