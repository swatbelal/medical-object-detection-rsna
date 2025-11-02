import torch
from ultralytics import YOLO


def yolo_model(model_str: str):
    model = YOLO(model_str)

    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    return model
