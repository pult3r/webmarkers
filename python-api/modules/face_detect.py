import uuid
import os
import cv2
import base64
import io
import numpy as np
from PIL import Image, ImageDraw
from fastapi import UploadFile
from typing import List
from lib.utils import encode_image_to_base64

# Wczytaj klasyfikator OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
IMAGE_OUTPUT_DIR = "static/face_detected"

os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

async def detect_faces(file: UploadFile, base_url: str):
    contents = await file.read()
    np_img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    pil_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    face_data = []
    for (x, y, w, h) in faces:
        draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
        face_data.append({"box": [int(x), int(y), int(w), int(h)]})

    # Zapisz i zakoduj obraz
    image_id = str(uuid.uuid4())
    output_path = os.path.join(IMAGE_OUTPUT_DIR, f"{image_id}.jpg")
    pil_image.save(output_path)

    image_base64 = encode_image_to_base64(pil_image)
    image_url = f"{base_url}/face-image/{image_id}"

    return {
        "faces": face_data,
        "image_url": image_url,
        "image_base64": image_base64
    }
