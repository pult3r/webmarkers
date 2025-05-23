# modules/clip_module.py

import os
import numpy as np
from PIL import Image
import torch
import faiss
import cv2
from torchvision import transforms
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
import clip as openai_clip
model_clip, preprocess_clip = openai_clip.load("ViT-B/32", device=device)

# Load YOLO model
model_yolo = YOLO("yolov8n.pt")

# FAISS index and metadata
index_path = "clip_index.index"
metadata_path = "clip_metadata.npy"

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = faiss.IndexFlatL2(512)

if os.path.exists(metadata_path):
    metadata = list(np.load(metadata_path, allow_pickle=True))
else:
    metadata = []

# Map YOLO class IDs to names (COCO dataset)
CLASS_NAMES = model_yolo.model.names

async def embed_clip_full_response(file):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    results = model_yolo.predict(rgb_img)
    detections = []
    embeddings_list = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        class_id = int(box.cls)
        conf = float(box.conf)
        name = CLASS_NAMES[class_id]

        roi = pil_img.crop((x1, y1, x2, y2))
        roi_processed = preprocess_clip(roi).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model_clip.encode_image(roi_processed).cpu().numpy()

        faiss_id = len(metadata)
        index.add(embedding.astype('float32'))
        metadata.append({
            "faiss_id": faiss_id,
            "box": [x1, y1, x2, y2],
            "class": class_id,
            "name": name,
            "conf": conf
        })

        detections.append({
            "box": [x1, y1, x2, y2],
            "class": class_id,
            "name": name,
            "confidence": conf,
            "embedding": embedding.tolist()[0],
            "faiss_id": faiss_id
        })

    # Save updated index and metadata
    faiss.write_index(index, index_path)
    np.save(metadata_path, np.array(metadata, dtype=object), allow_pickle=True)

    return {
        "detections": detections,
        "message": f"Added {len(detections)} new embeddings to FAISS index"
    }

def search_clip(image_path=None, text=None, top_k=5):
    if image_path:
        img = Image.open(image_path).convert("RGB")
        img_processed = preprocess_clip(img).unsqueeze(0).to(device)
        with torch.no_grad():
            query_embedding = model_clip.encode_image(img_processed).cpu().numpy()
    elif text:
        with torch.no_grad():
            query_embedding = model_clip.encode_text(openai_clip.tokenize([text]).to(device)).cpu().numpy()
    else:
        return {"error": "No input provided"}

    D, I = index.search(query_embedding.astype("float32"), top_k)
    results = []
    for idx in I[0]:
        results.append(metadata[idx])
    return results

def get_metadata_by_id(faiss_id):
    try:
        return metadata[int(faiss_id)]
    except:
        return {"error": "Invalid ID"}
