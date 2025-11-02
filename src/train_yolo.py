"""
train_yolo.py
Training runner for YOLOv8 using the RSNA processed dataset.
"""
from config import BASE_DIR
from tools import yolo_model

# data config YAML
DATA_YAML = BASE_DIR / "data.yaml"

# output directory for weights
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_yolo():
    model = yolo_model("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=5,
        patience=3,
        imgsz=512,
        batch=2,
        workers=2,
        project=str(OUTPUT_DIR),
        name="rsna_yolov8n"
    )


if __name__ == "__main__":
    train_yolo()
    print("training complete.")
