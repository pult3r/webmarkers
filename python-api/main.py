from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import shutil
import os
import uuid
from pathlib import Path
import io
import base64
from PIL import Image, ImageDraw
import numpy as np
import cv2

from fastapi.responses import FileResponse
from modules import detect, pose_detect
from fastapi import Request


from modules.color_detect import detect_dominant_colors
from modules.detect import run_detect
from modules.pose_detect import run_pose_detect
from modules.face_detect import detect_faces
from modules.describe import describe_image
from modules.clip_module import embed_clip_full_response, search_clip, get_metadata_by_id
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



@app.get("/image/color_previews/{filename}", tags=["Helpers"], summary="Color preview")
async def get_color_preview(filename: str):
    path = os.path.join("generated/color_previews", filename)
    return FileResponse(path)

@app.get("/image/{image_id}", tags=["Helpers"], summary="Show image")
async def get_image(image_id: str):
    path = f"saved_images/{image_id}.png"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return {"error": "Image not found"}

@app.get("/face-image/{image_id}", tags=["Helpers"], summary="Show image with faces")
async def get_image(image_id: str):
    path = f"static/face_detected/{image_id}.jpg"
    return FileResponse(path, media_type="image/jpeg")

@app.post("/detect", tags=["Detection"], summary="Object Detection: yolov8n.pt - bounding box, class, name, confidence, image_url, image_base64 [image contain red border]")
async def detect_objects(request: Request, file: UploadFile = File(...)):
    host = str(request.base_url).rstrip("/")
    return await detect.run_detect(file, host)

@app.post("/pose-detect", tags=["Detection"], summary="Pose Detection : yolov8n-pose.pt - bounding box, keypoints, class, name, confidence, image_url, image_base64 [image contain red border and keypoints]")
async def pose_objects(request: Request, file: UploadFile = File(...)):
    host = str(request.base_url).rstrip("/")
    return await pose_detect.run_pose_detect(file, host)

@app.post("/color-detect", tags=["Detection"] , summary="Image Main Colors Detection")
async def color_detect(file: UploadFile = File(...), num_colors: int = Query(3, ge=1, le=20)):
    return await detect_dominant_colors(file, num_colors)

@app.post("/face-detect", tags=["Face Detection"], summary="Face Detection : OpenCV - image_url, image_base64")
async def face_detect(file: UploadFile = File(...), request: Request = None):
    base_url = str(request.base_url).rstrip("/")
    result = await detect_faces(file, base_url)
    return result

@app.post("/face-recognize", tags=["Face Recognition"])
def face_recognize(file: UploadFile = File(...)):
    path = save_uploaded_image(file)
    return recognize_face(path)

@app.post("/describe", tags=["Image Description"])
def describe(file: UploadFile = File(...)):
    path = save_uploaded_image(file)
    return describe_image(path)

@app.post("/clip-embed", tags=["CLIP"])
async def clip_embed(file: UploadFile = File(...)):
    return await embed_clip_full_response(file)

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
