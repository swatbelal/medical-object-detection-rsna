"""
inference.py

Robust inference utility for YOLOv8 models (ultralytics).
Saves overlay images, JSON results, optional YOLO txt labels and cropped detections.

Usage examples (CLI):
    # basic (uses device auto-detect)
    python src/inference.py --weights models/rsna_yolov8n/weights/best.pt --source data/val/images --out results/inference_debug

    # low-confidence debugging (show more candidate boxes)
    python src/inference.py --weights models/rsna_yolov8n/weights/best.pt --source data/val/images --out results/inference_debug --conf 0.05 --debug

    # save crops + save JSON + save YOLO txt labels
    python src/inference.py --weights models/rsna_yolov8n/weights/best.pt --source data/val/images \
        --out results/inference_debug --save-crops --save-json --save-txt

Programmatic usage:
    from src.inference import run_inference
    results = run_inference(weights_path, images_dir, out_dir, conf=0.25, save_json=True)
"""

from pathlib import Path
import json
import time
import argparse
from typing import Union, List, Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Map class indices to names. RSNA has single class "pneumonia" -> index 0
CLASS_NAMES = {0: "pneumonia"}


def _draw_box(img: np.ndarray, box: List[float], label: str, conf: float):
    """
    Draw a single box with label and confidence on image (in-place).
    box: [x1,y1,x2,y2]
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img.shape[:2]

    # Thickness and font scale relative to image size
    thickness = max(1, int(round(0.002 * (w + h) / 2)))
    font_scale = max(0.4, (w + h) / 1000 * 0.4)
    color = (0, 220, 0)  # green

    # rectangle (filled background for text)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    caption = f"{label} {conf:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x1, y1 - text_h - baseline - 4), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(img, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def _save_yolo_txt(txt_path: Path, boxes: List[List[float]], classes: List[int], confs: List[float], img_size: tuple):
    """
    Save YOLO txt: class x_center y_center w h (normalized)
    boxes in absolute coordinates [x1,y1,x2,y2]
    img_size: (w,h)
    """
    w_img, h_img = img_size
    lines = []
    for box, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        xc = x1 + bw / 2
        yc = y1 + bh / 2
        lines.append(f"{cls} {xc / w_img:.6f} {yc / h_img:.6f} {bw / w_img:.6f} {bh / h_img:.6f}")
    txt_path.write_text("\n".join(lines))


def run_inference(
    weights_path: str,
    source: Union[str, Path],
    out_dir: Union[str, Path],
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = None,
    save_crops: bool = False,
    save_json: bool = True,
    save_txt: bool = False,
    max_det: int = 300,
    debug: bool = False,
) -> Dict[str, dict]:
    """
    Run inference on a folder or single image.

    Args:
        weights_path: path to .pt weights
        source: folder or image file
        out_dir: where to save overlays/json/txt/crops
        conf: confidence threshold for predictions (0-1)
        iou: NMS IoU threshold
        device: "cuda" or "cpu" or None (auto)
        save_crops: save each detection crop (per-image subfolder "crops")
        save_json: save detections JSON (results.json)
        save_txt: save one YOLO-format .txt per image
        max_det: max detections per image
        debug: if True, lowers logging threshold and saves raw results with conf=0.01
    Returns:
        results_all: dict keyed by image filename with detection arrays and metadata
    """
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights_path)).to(device)

    # For debug we will run a secondary pass with low conf to save more candidates optionally
    conf_run = 0.01 if debug else conf

    # Collect images
    if source.is_dir():
        image_paths = sorted([p for p in source.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]])
    elif source.is_file():
        image_paths = [source]
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {source}")

    # output subfolders
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    crops_dir = out_dir / "crops" if save_crops else None
    if crops_dir:
        crops_dir.mkdir(exist_ok=True)
    json_path = out_dir / "results.json"
    txt_dir = out_dir / "yolo_txt" if save_txt else None
    if txt_dir:
        txt_dir.mkdir(exist_ok=True)

    results_all = {}
    t0 = time.time()

    for img_path in image_paths:
        t_img0 = time.time()
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot read {img_path}, skipping")
            continue
        h, w = img.shape[:2]

        # Run model (we pass conf_run)
        res = model.predict(str(img_path), conf=conf_run, iou=iou, max_det=max_det, verbose=False)

        # ultralytics returns list of Result; take first
        r = res[0]

        # raw boxes / conf / cls arrays
        if len(r.boxes) == 0:
            boxes = np.zeros((0, 4), dtype=float)
            confs = np.array([])
            classes = np.array([], dtype=int)
        else:
            boxes = r.boxes.xyxy.cpu().numpy()  # Nx4
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

        # filter boxes by conf threshold (main threshold)
        keep_mask = confs >= conf
        filtered_boxes = boxes[keep_mask]
        filtered_confs = confs[keep_mask]
        filtered_classes = classes[keep_mask]

        # draw overlay from filtered boxes
        overlay = img.copy()
        for box, cls, c in zip(filtered_boxes, filtered_classes, filtered_confs):
            label = CLASS_NAMES.get(int(cls), str(int(cls)))
            _draw_box(overlay, box.tolist(), label, float(c))

        # save overlay image
        out_img_path = overlays_dir / img_path.name
        cv2.imwrite(str(out_img_path), overlay)

        # save crops if requested (use filtered boxes)
        if save_crops and len(filtered_boxes) > 0:
            img_crops_dir = crops_dir / img_path.stem
            img_crops_dir.mkdir(parents=True, exist_ok=True)
            for i, (box, cls, c) in enumerate(zip(filtered_boxes, filtered_classes, filtered_confs)):
                x1, y1, x2, y2 = [int(v) for v in box]
                crop = img[y1:y2, x1:x2]
                crop_name = img_crops_dir / f"{img_path.stem}_det{i}_{int(c*100)}.png"
                cv2.imwrite(str(crop_name), crop)

        # save YOLO txt (normalized)
        if save_txt and len(filtered_boxes) > 0:
            _save_yolo_txt(txt_dir / f"{img_path.stem}.txt", filtered_boxes.tolist(), filtered_classes.tolist(), filtered_confs.tolist(), (w, h))

        # collect per-image JSON-like structure (both filtered and raw)
        detections = []
        for (box, cls, c) in zip(filtered_boxes.tolist(), filtered_classes.tolist(), filtered_confs.tolist()):
            x1, y1, x2, y2 = box
            detections.append({
                "class_id": int(cls),
                "class_name": CLASS_NAMES.get(int(cls), str(int(cls))),
                "conf": float(c),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "area": float((x2 - x1) * (y2 - y1)),
            })

        results_all[img_path.name] = {
            "width": int(w),
            "height": int(h),
            "n_pred": int(len(filtered_boxes)),
            "predictions": detections,
            "raw_count": int(len(boxes)),
        }

        # debug / logging
        dt = time.time() - t_img0
        print(f"[{img_path.name}] raw={len(boxes)} kept={len(filtered_boxes)} time={dt:.3f}s")

    total_time = time.time() - t0
    print(f"Processed {len(results_all)} images in {total_time:.2f}s ({total_time/len(results_all):.3f}s/img)")

    if save_json:
        json_path.write_text(json.dumps(results_all, indent=2))

    return results_all


# CLI wrapper
def _parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 inference utility (debug & production)")
    p.add_argument("--weights", required=True, help="Path to .pt weights")
    p.add_argument("--source", required=True, help="Image file or directory")
    p.add_argument("--out", default="results/inference", help="Output folder")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device", default=None, help="cuda or cpu (auto detect if not set)")
    p.add_argument("--save-crops", action="store_true", help="Save cropped detections")
    p.add_argument("--save-json", action="store_true", help="Save results.json")
    p.add_argument("--save-txt", action="store_true", help="Save YOLO txt per image")
    p.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    p.add_argument("--debug", action="store_true", help="Debug mode (low conf pass + verbose)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    res = run_inference(
        weights_path=args.weights,
        source=args.source,
        out_dir=args.out,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_crops=args.save_crops,
        save_json=args.save_json,
        save_txt=args.save_txt,
        max_det=args.max_det,
        debug=args.debug,
    )
    print("Done. Summary:")
    # pretty print a compact summary
    for img, info in list(res.items())[:10]:
        print(f"  {img}: {info['n_pred']} detections")
