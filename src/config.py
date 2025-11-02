from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
RESULT_DIR = BASE_DIR / "results"

# --- Raw RSNA dicoms
DICOM_TRAIN_DIR = DATA_DIR / "stage_2_train_images"
DICOM_TEST_DIR = DATA_DIR / "stage_2_test_images"

# --- PNGs
PNG_TRAIN_DIR = PROC_DIR / "train_png"
PNG_TEST_DIR = PROC_DIR / "test_png"

# --- YOLO labels
LABELS_DIR = RESULT_DIR / "train_labels"

# --- final split
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

TRAIN_IMAGES_DIR = TRAIN_DIR / "images"
TRAIN_LABELS_DIR = TRAIN_DIR / "labels"
VAL_IMAGES_DIR = VAL_DIR / "images"
VAL_LABELS_DIR = VAL_DIR / "labels"

TRAIN_RATIO = 0.8

ALL_PATHS = {
    name: value for name, value in globals().items()
    if name.isupper() and isinstance(value, Path)
}

if __name__ == "__main__":
    for k, v in ALL_PATHS.items():
        print(f"{k:20s} -> {v}")
