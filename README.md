# BGU-IDL-img2gps
Image-to-GPS regression on university campus: training and evaluating models that map images to precise geocoordinates.

## Data
CSV must include: `image_id`, `sector_label`, `lat`, `lon`.

## Training (new models)
Coordinate regression (DINOv2 ViT-B/14 + regression head):
```bash
python -m src.train_tasks --task coords --csv-path data/metadata1.csv --img-dir data/images --pretrained --amp
```

Region classification (Swin Base):
```bash
python -m src.train_tasks --task region --csv-path data/metadata1.csv --img-dir data/images --pretrained --amp --randaugment
```

Notes:
- `sector_label` is auto-factorized to `0..N-1`.
- `--coord-mode latlon` + `--coord-norm standard` enables statistical normalization for regression.
