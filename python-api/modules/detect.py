import uuid
import base64
import os
from fastapi import UploadFile
import io
from PIL import Image, ImageDraw
import numpy as np
import cv2

from .clip_module import model_yolo, CLASS_NAMES

OUTPUT_DIR = "saved_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def run_detect(file: UploadFile, host: str):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    results = model_yolo.predict(rgb_img)
    draw = ImageDraw.Draw(pil_img)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        class_id = int(box.cls)
        conf = float(box.conf)
        name = CLASS_NAMES[class_id]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        detections.append({
            "box": [x1, y1, x2, y2],
            "class": class_id,
            "name": name,
            "confidence": conf
        })

    # Save to disk
    image_id = str(uuid.uuid4())
    file_path = os.path.join(OUTPUT_DIR, f"{image_id}.png")
    pil_img.save(file_path, format="PNG")

    # Convert to base64
    with open(file_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    return {
        "detections": detections,
        "image_url": f"{host}/image/{image_id}",
        "image_base64": image_base64
    }
