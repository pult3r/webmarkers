
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import shutil
import os
import uuid
from pathlib import Path

from modules.detect import detect_objects
from modules.pose import detect_pose
from modules.color import detect_dominant_colors
from modules.face import detect_faces, recognize_face
from modules.describe import describe_image
from modules.clip_module import embed_clip, search_clip, get_metadata_by_id
from modules.semantic import get_semantic_description
from modules.visualize import visualize_image_by_id, visualize_base64_by_id

app = FastAPI()

# Allow CORS for all origins (customize for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------- IMAGE UTILS --------------------
def save_uploaded_image(file: UploadFile) -> str:
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

# -------------------- ENDPOINTS --------------------

@app.post("/detect", tags=["Object Detection"])
def detect(file: UploadFile = File(...)):
    path = save_uploaded_image(file)
    return detect_objects(path)

@app.post("/pose-detect", tags=["Pose Detection"])
def pose_detect(file: UploadFile = File(...)):
    path = save_uploaded_image(file)
    return detect_pose(path)

@app.post("/color-detect", tags=["Color Detection"])
def color_detect(file: UploadFile = File(...)):
    path = save_uploaded_image(file)
    return detect_dominant_colors(path)

@app.post("/face-detect", tags=["Face Detection"])
def face_detect(file: UploadFile = File(...)):
    path = save_uploaded_image(file)
    return detect_faces(path)

@app.post("/face-recognize", tags=["Face Recognition"])
def face_recognize(file: UploadFile = File(...)):
    path = save_uploaded_image(file)
    return recognize_face(path)

@app.post("/describe", tags=["Image Description"])
def describe(file: UploadFile = File(...)):
    path = save_uploaded_image(file)
    return describe_image(path)

@app.post("/clip-embed", tags=["CLIP"])
def clip_embed(file: UploadFile = File(...)):
    path = save_uploaded_image(file)
    return embed_clip(path)

@app.post("/clip-search", tags=["CLIP"])
def clip_search(file: UploadFile = File(...)):
    path = save_uploaded_image(file)
    return search_clip(path)

@app.post("/text-search", tags=["CLIP"])
def text_search(query: str = Form(...)):
    return search_clip(text=query)

@app.get("/clip-metadata/{id}", tags=["CLIP"])
def clip_metadata(id: str):
    return get_metadata_by_id(id)

@app.get("/visualize/{id}", tags=["Visualization"])
def visualize(id: str):
    return visualize_image_by_id(id)

@app.get("/visualize-base64/{id}", tags=["Visualization"])
def visualize_base64(id: str):
    return visualize_base64_by_id(id)

@app.get("/semantic/{id}", tags=["Semantic Description"])
def semantic(id: str):
    return get_semantic_description(id)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
