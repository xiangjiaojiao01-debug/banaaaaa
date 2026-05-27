from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

LOCAL_MODEL_PATH = Path(r"C:\Users\user\Desktop\yolo\banana_yolo\v19\weights\best.pt")
MODEL_PATH = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH.exists() else Path("best.pt")
SOURCE_DIR = Path(r"C:\Users\user\Desktop\dataset(沒有甜度版)_cropped")
OUTPUT_DIR = Path(r"C:\Users\user\Desktop\yolo\banana_predict_result")
DEFAULT_CONF = 0.20
S_CONF = 0.20

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


def filter_yolo_detections(detections):
    all_detections = [d for d in detections if d["label"] == "all"]
    s_detections = [d for d in detections if d["label"] == "s" and d["confidence"] >= S_CONF]

    if not all_detections:
        return None, s_detections

    best_all = max(all_detections, key=lambda d: d["confidence"])
    all_box = best_all["box"]
    s_detections = [d for d in s_detections if _inside(d["box"], all_box)]

    return best_all, s_detections


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

    return [best_all] + s_detections


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



