"""
data_utils.py
=============
Utility functions for RSNA Pneumonia Detection project.

Includes:
- DICOM → PNG conversion
- CSV → YOLO label generation
- Train/val split
- CLI (__main__) entry point
"""

import sys
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import pandas as pd
import pydicom

import config


# ------------------------------------------------------------
# DICOM → PNG (multithreaded)
# ------------------------------------------------------------
def _dicom_to_png_single(d: Path, output_dir: Path):
    out_path = output_dir / f"{d.stem}.png"

    # if PNG exists already → skip to save time
    if out_path.exists():
        return

    try:
        ds = pydicom.dcmread(str(d))
        arr = ds.pixel_array
        cv2.imwrite(str(out_path), arr)
    except Exception as e:
        print(f"[ERROR] {d.name}: {e}")


def convert_dicom_to_png(input_dir: Path, output_dir: Path, workers: int = 8):
    """
    Convert all DICOM (.dcm) files to PNGs with the same filename (patientId).

    Multithreaded.
    workers: number of threads (8 is a good default)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dicoms = list(input_dir.glob("*.dcm"))

    print(f"Converting {len(dicoms)} DICOMs → PNG using {workers} threads...")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_dicom_to_png_single, d, output_dir): d for d in dicoms}
        for f in as_completed(futures):
            _ = f
            pass  # we don't need per-file prints (too much noise)

    # print("✅ DICOM → PNG conversion complete.")


# ------------------------------------------------------------
# CSV → YOLO labels
# ------------------------------------------------------------
def csv_to_yolo(csv_path: Path, images_dir: Path, labels_out_dir: Path):
    """
    Convert RSNA stage_2_train_labels.csv to YOLO format.

    YOLO txt format:
        class x_center y_center w h   (normalized)

    class = 0 (pneumonia)
    """
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # keep only pneumonia positive rows
    df = df[df["Target"] == 1]

    grouped = df.groupby("patientId")

    for pid, rows in grouped:
        img_path = images_dir / f"{pid}.png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        lines = []

        for _, r in rows.iterrows():
            x = r["x"]
            y = r["y"]
            bw = r["width"]
            bh = r["height"]

            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            w_norm = bw / w
            h_norm = bh / h

            lines.append(f"0 {x_center} {y_center} {w_norm} {h_norm}")

        (labels_out_dir / f"{pid}.txt").write_text("\n".join(lines))


# ------------------------------------------------------------
# Train/val split (COPY)
# ------------------------------------------------------------
def split_train_val():
    """
    Split PNG + YOLO labels into train/val folders using ratio in config.TRAIN_RATIO.
    Uses COPY (files remain in processed folder).
    """
    cfg = config  # shorthand

    cfg.TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.VAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.VAL_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    pngs = sorted(list(cfg.PNG_TRAIN_DIR.glob("*.png")))
    total = len(pngs)
    split_index = int(total * cfg.TRAIN_RATIO)

    train_set = pngs[:split_index]
    val_set = pngs[split_index:]

    for p in train_set:
        pid = p.stem
        lbl = cfg.LABELS_DIR / f"{pid}.txt"
        shutil.copy(p, cfg.TRAIN_IMAGES_DIR / p.name)
        if lbl.exists():
            shutil.copy(lbl, cfg.TRAIN_LABELS_DIR / lbl.name)

    for p in val_set:
        pid = p.stem
        lbl = cfg.LABELS_DIR / f"{pid}.txt"
        shutil.copy(p, cfg.VAL_IMAGES_DIR / p.name)
        if lbl.exists():
            shutil.copy(lbl, cfg.VAL_LABELS_DIR / lbl.name)


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/data_utils.py [convert|labels|split]")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "convert":
        convert_dicom_to_png(config.DICOM_TRAIN_DIR, config.PNG_TRAIN_DIR)
        convert_dicom_to_png(config.DICOM_TEST_DIR, config.PNG_TEST_DIR)
        print("✅ DICOM → PNG conversion complete.")

    elif cmd == "labels":
        csv_to_yolo(
            csv_path=config.BASE_DIR / "data" / "stage_2_train_labels.csv",
            images_dir=config.PNG_TRAIN_DIR,
            labels_out_dir=config.LABELS_DIR,
        )
        print("✅ CSV → YOLO label generation complete.")

    elif cmd == "split":
        split_train_val()
        print("✅ Train/val split complete.")

    else:
        print(f"Unknown command: {cmd}")
