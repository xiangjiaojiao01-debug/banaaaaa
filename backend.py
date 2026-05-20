from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"
SOURCE_DIR = BASE_DIR / "dataset_cropped"
OUTPUT_DIR = Path("/tmp/banana_predict_result")
DEFAULT_CONF = 0.20
S_CONF = 0.20
DARK_MIN_AREA = 700
DARK_IOU_SKIP = 0.25

model = YOLO(str(MODEL_PATH))

CLASS_COLORS = {
    "all": "white",
    "s": "blue",
}


def _center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _inside(inner_box, outer_box):
    cx, cy = _center(inner_box)
    ox1, oy1, ox2, oy2 = outer_box
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0


def _clip_box(box, width, height):
    x1, y1, x2, y2 = box
    return [
        max(0, min(float(x1), width - 1)),
        max(0, min(float(y1), height - 1)),
        max(0, min(float(x2), width - 1)),
        max(0, min(float(y2), height - 1)),
    ]


def filter_yolo_detections(detections):
    all_detections = [d for d in detections if d["label"] == "all"]
    s_detections = [d for d in detections if d["label"] == "s" and d["confidence"] >= S_CONF]

    if not all_detections:
        return None, s_detections

    best_all = max(all_detections, key=lambda d: d["confidence"])
    all_box = best_all["box"]
    s_detections = [d for d in s_detections if _inside(d["box"], all_box)]

    return best_all, s_detections


def detect_dark_spots(image_path, all_box, existing_s):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        rgb = np.array(image)

    height, width = rgb.shape[:2]
    all_box = _clip_box(all_box, width, height)
    ax1, ay1, ax2, ay2 = [int(v) for v in all_box]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    dark_mask = cv2.inRange(hsv, np.array([0, 45, 0]), np.array([35, 255, 115]))
    brown_mask = cv2.inRange(hsv, np.array([5, 35, 20]), np.array([35, 255, 150]))
    mask = cv2.bitwise_or(dark_mask, brown_mask)

    inside_all = np.zeros(mask.shape, dtype=np.uint8)
    inside_all[ay1:ay2, ax1:ax2] = 255
    mask = cv2.bitwise_and(mask, inside_all)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < DARK_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        fill_ratio = area / max(w * h, 1)
        if aspect_ratio > 6 or fill_ratio < 0.18:
            continue

        box = [float(x), float(y), float(x + w), float(y + h)]
        if not _inside(box, all_box):
            continue
        if any(_iou(box, det["box"]) > DARK_IOU_SKIP for det in existing_s):
            continue

        spots.append({
            "label": "s",
            "confidence": 0.20,
            "box": box,
            "source": "dark_fallback",
        })

    return spots


def predict_image(image_path, conf=DEFAULT_CONF):
    result = model.predict(source=str(image_path), conf=conf, save=False, verbose=False)[0]

    raw_detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = str(model.names[cls_id])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

        if label not in {"all", "s"}:
            continue

        raw_detections.append({
            "label": label,
            "confidence": confidence,
            "box": [x1, y1, x2, y2],
            "source": "yolo",
        })

    best_all, s_detections = filter_yolo_detections(raw_detections)
    if best_all is None:
        return s_detections

    fallback_spots = detect_dark_spots(image_path, best_all["box"], s_detections)
    return [best_all] + s_detections + fallback_spots


def draw_detections(image_path, detections, output_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)

        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["box"]
            color = CLASS_COLORS.get(label, "red")
            text = f"{label} {conf:.2f}"

            width = 3 if label == "all" else 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            text_box = draw.textbbox((x1, y1), text)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            text_y = max(0, y1 - text_h - 4)
            draw.rectangle([x1, text_y, x1 + text_w + 6, text_y + text_h + 4], fill=color)
            draw.text((x1 + 3, text_y + 2), text, fill="black")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)


def run_folder(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR, conf=DEFAULT_CONF):
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG"]

    images = []
    for ext in image_exts:
        images.extend(source_dir.rglob(ext))

    print(f"模型：{MODEL_PATH}")
    print(f"讀取圖片：{source_dir}")
    print(f"輸出結果：{output_dir}")
    print(f"找到圖片數量：{len(images)}")

    for image_path in images:
        detections = predict_image(image_path, conf=conf)
        relative_path = image_path.relative_to(source_dir)
        output_path = output_dir / relative_path
        draw_detections(image_path, detections, output_path)

        all_count = sum(1 for d in detections if d["label"] == "all")
        s_count = sum(1 for d in detections if d["label"] == "s")
        print(f"完成：{relative_path}，all={all_count}，s={s_count}")


if __name__ == "__main__":
    run_folder()




