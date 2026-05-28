from pathlib import Path
import colorsys

from PIL import Image, ImageDraw
from ultralytics import YOLO

LOCAL_MODEL_PATH = Path(r"C:\Users\user\Desktop\yolo\banana_yolo\v19\weights\best.pt")
MODEL_PATH = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH.exists() else Path("best.pt")
SOURCE_DIR = Path(r"C:\Users\user\Desktop\dataset(沒有甜度版)_cropped")
OUTPUT_DIR = Path(r"C:\Users\user\Desktop\yolo\banana_predict_result")
DEFAULT_CONF = 0.20
S_CONF = 0.20
LOW_SPOT_MAX = 5.0
MID_SPOT_MAX = 15.0
MAX_SPOT_BOX_AREA_RATIO = 0.25
YELLOW_RATIO_MIN = 0.20
GREEN_RATIO_MIN = 0.20

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


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _clip_box(box, outer_box):
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = outer_box
    return [
        max(x1, ox1),
        max(y1, oy1),
        min(x2, ox2),
        min(y2, oy2),
    ]


def _clip_box_to_image(box, image_size):
    width, height = image_size
    return [
        max(0, min(width, int(box[0]))),
        max(0, min(height, int(box[1]))),
        max(0, min(width, int(box[2]))),
        max(0, min(height, int(box[3]))),
    ]


def _union_box_area(boxes):
    x_edges = sorted({x for box in boxes for x in (box[0], box[2])})
    total = 0.0

    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue

        intervals = []
        for x1, y1, x2, y2 in boxes:
            if x1 <= left and right <= x2 and y2 > y1:
                intervals.append((y1, y2))

        intervals.sort()
        merged = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        total += (right - left) * sum(end - start for start, end in merged)

    return total


def analyze_banana_color(image_path, detections):
    all_box = next((d for d in detections if d["label"] == "all"), None)
    if all_box is None:
        return {
            "banana_color": "不明",
            "yellow_ratio": 0.0,
            "green_ratio": 0.0,
        }

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        crop_box = _clip_box_to_image(all_box["box"], image.size)
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return {
                "banana_color": "不明",
                "yellow_ratio": 0.0,
                "green_ratio": 0.0,
            }

        crop = image.crop(crop_box)
        crop.thumbnail((220, 220))

        total = max(1, crop.size[0] * crop.size[1])
        yellow_count = 0
        green_count = 0
        brown_spot_count = 0

        for red, green, blue in crop.getdata():
            hue, saturation, value = colorsys.rgb_to_hsv(
                red / 255,
                green / 255,
                blue / 255,
            )
            hue_degrees = hue * 360

            is_green = 70 < hue_degrees <= 170 and saturation >= 0.20 and value >= 0.25
            is_yellow = 28 <= hue_degrees <= 70 and saturation >= 0.25 and value >= 0.35
            is_brown_spot = (
                (
                    10 <= hue_degrees <= 55
                    and saturation >= 0.18
                    and 0.12 <= value <= 0.62
                )
                or (value < 0.22 and saturation >= 0.12)
            ) and not is_green

            if is_brown_spot:
                brown_spot_count += 1
            elif is_yellow:
                yellow_count += 1
            elif is_green:
                green_count += 1

    yellow_ratio = yellow_count / total
    green_ratio = green_count / total
    color_spot_base = max(1, yellow_count + brown_spot_count)
    color_black_spot_pct = brown_spot_count / color_spot_base * 100

    if yellow_ratio >= YELLOW_RATIO_MIN:
        banana_color = "偏黃"
    elif green_ratio >= GREEN_RATIO_MIN:
        banana_color = "偏綠"
    else:
        banana_color = "不明"

    return {
        "banana_color": banana_color,
        "yellow_ratio": round(yellow_ratio, 3),
        "green_ratio": round(green_ratio, 3),
        "color_black_spot_pct": round(color_black_spot_pct, 2),
    }


def filter_yolo_detections(detections):
    all_detections = [d for d in detections if d["label"] == "all"]
    s_detections = [d for d in detections if d["label"] == "s" and d["confidence"] >= S_CONF]

    if not all_detections:
        return None, s_detections

    best_all = max(
        all_detections,
        key=lambda d: (
            sum(1 for spot in s_detections if _inside(spot["box"], d["box"])),
            d["confidence"],
        ),
    )
    all_box = best_all["box"]
    banana_area = _box_area(all_box)
    s_detections = [
        d for d in s_detections
        if _inside(d["box"], all_box)
        and _box_area(_clip_box(d["box"], all_box)) / banana_area <= MAX_SPOT_BOX_AREA_RATIO
    ]

    return best_all, s_detections


def calculate_black_spot_pct(detections):
    all_box = next((d for d in detections if d["label"] == "all"), None)
    s_detections = [d for d in detections if d["label"] == "s"]

    if all_box is None:
        return 0.0

    banana_area = _box_area(all_box["box"])
    if banana_area <= 0:
        return 0.0

    spot_boxes = [
        _clip_box(d["box"], all_box["box"])
        for d in s_detections
    ]
    spot_boxes = [box for box in spot_boxes if _box_area(box) > 0]
    spot_area = _union_box_area(spot_boxes)
    return round(min(spot_area / banana_area * 100, 100.0), 2)


def classify_yolo_black_spot(black_spot_pct, banana_color="不明"):
    if black_spot_pct < LOW_SPOT_MAX:
        if banana_color == "偏黃":
            return {
                "black_spot_index": "低",
                "standard": "YOLO <5% 且顏色偏黃 = 低黑斑但已轉黃",
                "ripeness": "已轉黃或剛熟",
                "sweetness_level": "甜度中等",
            }
        return {
            "black_spot_index": "低",
            "standard": "YOLO <5% = 低",
            "ripeness": "偏生或剛熟",
            "sweetness_level": "不甜或甜度偏低",
        }
    if black_spot_pct < MID_SPOT_MAX:
        return {
            "black_spot_index": "中",
            "standard": "YOLO 5-15% = 中",
            "ripeness": "成熟中",
            "sweetness_level": "甜度中等",
        }
    return {
        "black_spot_index": "高",
        "standard": "YOLO >=15% = 高",
        "ripeness": "成熟或偏熟",
        "sweetness_level": "甜度較高",
    }


def analyze_yolo_result(detections, image_path=None):
    yolo_black_spot_pct = calculate_black_spot_pct(detections)
    s_count = sum(1 for d in detections if d["label"] == "s")
    all_count = sum(1 for d in detections if d["label"] == "all")
    color_info = (
        analyze_banana_color(image_path, detections)
        if image_path is not None
        else {
            "banana_color": "不明",
            "yellow_ratio": 0.0,
            "green_ratio": 0.0,
            "color_black_spot_pct": 0.0,
        }
    )
    black_spot_pct = max(yolo_black_spot_pct, color_info["color_black_spot_pct"])
    result = classify_yolo_black_spot(black_spot_pct, color_info["banana_color"])
    if color_info["color_black_spot_pct"] > yolo_black_spot_pct:
        result["standard"] = f"顏色褐斑補償 {color_info['color_black_spot_pct']:.2f}%"

    result.update({
        "black_spot_pct": black_spot_pct,
        "yolo_black_spot_pct": yolo_black_spot_pct,
        "spot_count": s_count,
        "banana_count": all_count,
        "method": "YOLO",
        **color_info,
    })
    return result


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


def predict_and_analyze(image_path, conf=DEFAULT_CONF):
    detections = predict_image(image_path, conf=conf)
    analysis = analyze_yolo_result(detections, image_path=image_path)
    return detections, analysis


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



