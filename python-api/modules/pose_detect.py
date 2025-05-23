import uuid
import base64
import os
import io
from fastapi import UploadFile
from PIL import Image, ImageDraw
import numpy as np
import cv2
from ultralytics import YOLO

# Ścieżka zapisu obrazów
OUTPUT_DIR = "saved_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Wczytaj model do wykrywania sylwetek
model_pose = YOLO("yolov8n-pose.pt")

async def run_pose_detect(file: UploadFile, host: str):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    results = model_pose.predict(rgb_img)
    draw = ImageDraw.Draw(pil_img)
    detections = []

    for result in results:
        for box, keypoints in zip(result.boxes, result.keypoints.xy):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls)
            conf = float(box.conf)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Rysuj punkty ciała
            for point in keypoints.cpu().numpy():
                px, py = map(int, point)
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="blue")

            detections.append({
                "box": [x1, y1, x2, y2],
                "class": class_id,
                "confidence": conf,
                "keypoints": keypoints.cpu().numpy().tolist()
            })

    # Zapisz obraz z ramkami i punktami
    image_id = str(uuid.uuid4())
    file_path = os.path.join(OUTPUT_DIR, f"{image_id}.png")
    pil_img.save(file_path, format="PNG")

    with open(file_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    return {
        "detections": detections,
        "image_url": f"{host}/image/{image_id}",
        "image_base64": image_base64
    }
