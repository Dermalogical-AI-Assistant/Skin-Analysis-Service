from app.ml_models.yolo.loader import load_yolov9_model
from app.core import config
from PIL import Image
import io
import uuid
import os
import cloudinary.uploader
import cv2
import numpy as np
import random
import json

# Load YOLO model
model, classes = load_yolov9_model()

# Tạo màu ngẫu nhiên cố định cho mỗi label
label_to_color = {}

def get_color_for_label(label):
    if label not in label_to_color:
        label_to_color[label] = tuple(random.randint(100, 255) for _ in range(3))
    return label_to_color[label]

def draw_boxes_only(results, original_image, conf_threshold=0.5):
    img = np.array(original_image)
    boxes = results[0].boxes
    names = model.names

    for box in boxes:
        confidence = float(box.conf[0])
        if confidence < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = names[cls_id]
        color = get_color_for_label(label)

        # Vẽ bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)

        # # Kích thước label
        # (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # top_left = (x1, y1 - text_height - 10)
        # bottom_right = (x1 + text_width + 4, y1)

        # # Vẽ nền label (rectangle)
        # cv2.rectangle(img, top_left, bottom_right, color, thickness=cv2.FILLED)
        #
        # # Vẽ chữ trên nền
        # cv2.putText(img, label, (x1 + 2, y1 - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img

def detect_objects_yolov9(image_bytes: bytes, conf_threshold: float = 0.5):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(image, imgsz=640)

    # Vẽ box và label
    result_img = draw_boxes_only(results, image, conf_threshold=conf_threshold)

    # Convert về PIL để lưu
    img_pil = Image.fromarray(result_img)

    # Tạo thư mục tạm và lưu file
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(root_dir, "temp_images")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(save_dir, filename)
    img_pil.save(filepath)
    image_url = ''
    try:
        # Upload lên Cloudinary
        upload_result = cloudinary.uploader.upload(filepath, folder="yolov9_detections")
        image_url = upload_result.get("secure_url")
    except Exception as e:
        print(e)

    # Xoá file tạm
    os.remove(filepath)

    result_json_str = results[0].to_json()
    result_json = json.loads(result_json_str)

    filtered_result = []

    for item in result_json:
        if float(item.get("confidence", 0)) >= conf_threshold:
            label = item.get("name")
            item["color"] = get_color_for_label(label)
            filtered_result.append(item)

    return filtered_result,conf_threshold , image_url, classes
