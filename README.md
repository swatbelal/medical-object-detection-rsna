# rsna-detect

**Transfer learning–based medical object detection pipeline (skeleton)**  
RSNA Pneumonia Detection — project skeleton to build an end-to-end detection workflow using YOLOv8 and/or Faster R-CNN.

## Repo structure (placeholders)

```
rsna-detect/
├─ data/
│  ├─ raw/                 # raw downloaded files (Kaggle)
│  ├─ processed/           # processed/converted images and annotations
│  ├─ train/
│  │  ├─ images/
│  │  └─ labels/
│  └─ val/
│     ├─ images/
│     └─ labels/
├─ notebooks/
│  └─ object_detection_yolo.ipynb
├─ src/
│  ├─ data_utils.py        # (empty) utilities for conversion, parsing CSV -> COCO/YOLO
│  ├─ train_yolo.py        # (empty) training script for YOLOv8
│  ├─ train_detectron2.py  # (empty) optional Detectron2/Faster R-CNN trainer
│  ├─ inference.py         # (empty) inference and visualization utilities
│  └─ eval.py              # (empty) evaluation helpers (COCO/mAP)
├─ models/                 # saved checkpoints
├─ results/                # metrics, PR curves, sample predictions
├─ docs/
│  └─ architecture.md      # high level architecture / notes
├─ scripts/
│  ├─ download_data.sh     # placeholder script to download dataset (Kaggle steps)
│  └─ prepare_dataset.sh   # placeholder script for dataset conversion
├─ tests/
│  └─ test_data_utils.py   # (empty) unit tests skeleton
├─ README.md
├─ requirements.txt
├─ data.yaml               # YOLO data config (placeholder)
├─ .gitignore
└─ LICENSE
```

## Next steps
1. Populate `scripts/download_data.sh` with Kaggle commands or dataset instructions.
2. Implement `src/data_utils.py` functions to convert RSNA CSV annotations into YOLO/COCO formats.
3. Add training commands to `src/train_yolo.py` and `src/train_detectron2.py`.
4. Flesh out the notebook `notebooks/object_detection_yolo.ipynb` with runnable cells and example images.

## Notes
- This repository intentionally contains **no implementation code**. It's a skeleton to iterate on file-by-file as requested.
- After you review the structure I can start filling specific files per your instruction.
