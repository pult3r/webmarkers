import io
import uuid
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from fastapi import UploadFile
from fastapi.responses import FileResponse
import base64

IMAGE_SAVE_DIR = "generated/color_previews"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))

def generate_color_preview(colors):
    block_size = 50
    width = block_size * len(colors)
    image = Image.new("RGB", (width, block_size))
    for i, color in enumerate(colors):
        block = Image.new("RGB", (block_size, block_size), tuple(map(int, color)))
        image.paste(block, (i * block_size, 0))
    return image

async def detect_dominant_colors(file: UploadFile, num_colors: int = 3, server_address: str = "http://localhost:8000"):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))

    img_flat = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, n_init="auto")
    kmeans.fit(img_flat)
    colors = kmeans.cluster_centers_

    hex_colors = [rgb_to_hex(color) for color in colors]

    # Generowanie obrazka podglądowego
    preview_img = generate_color_preview(colors)
    img_id = str(uuid.uuid4())
    img_path = f"{IMAGE_SAVE_DIR}/{img_id}.png"
    preview_img.save(img_path)

    buffered = io.BytesIO()
    preview_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "hex_colors": hex_colors,
        "image_base64": img_base64,
        "image_url": f"{server_address}/image/color_previews/{img_id}.png"
    }
