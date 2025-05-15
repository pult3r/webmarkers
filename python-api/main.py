from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import face_recognition
import pickle
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import io
from sklearn.cluster import KMeans
from collections import Counter


# Inicjalizacja FastAPI
app = FastAPI()

# YOLO
model = YOLO("yolov8n.pt")
model_pose = YOLO("yolov8n-pose.pt")      # do detekcji sylwetki

# BLIP-2
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model_blip = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model_blip.to(device)

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

# Znane twarze
with open("face_encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(image)
    return {"detections": results[0].tojson()}

@app.post("/pose-detect")
async def pose_detect(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model_pose.predict(image)
    
    keypoints_data = []
    if results[0].keypoints:
        for person in results[0].keypoints.xy.cpu().numpy():
            keypoints = [{"x": float(p[0]), "y": float(p[1])} for p in person]
            keypoints_data.append(keypoints)
    
    return {"poses": keypoints_data}

@app.post("/color-detect")
async def color_detect(file: UploadFile = File(...), num_colors: int = 5):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(img_rgb)

    counter = Counter(kmeans.labels_)
    center_colors = kmeans.cluster_centers_

    # Posortuj kolory wg liczności występowania (najwięcej pikseli)
    #ranked_colors = [tuple(map(int, center_colors[i])) for i, _ in counter.most_common(num_colors)]

    ranked_colors = [
        rgb_to_hex(tuple(map(int, center_colors[i])))
        for i, _ in counter.most_common(num_colors)
    ]


    return {"dominant_colors_rgb": ranked_colors}

@app.post("/face-detect")
async def face_detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_img)

    faces = [
        {"top": top, "right": right, "bottom": bottom, "left": left}
        for top, right, bottom, left in face_locations
    ]

    return {"faces": faces}

@app.post("/face-recognize")
async def face_recognize(file: UploadFile = File(...), tolerance: float = 0.6):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    if not face_encodings:
        raise HTTPException(status_code=404, detail="No face found.")

    results = []
    for face_encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(
            [f["encoding"] for f in known_faces], face_encoding, tolerance
        )
        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_faces[match_index]["name"]

        results.append({
            "name": name,
            "location": {
                "top": location[0],
                "right": location[1],
                "bottom": location[2],
                "left": location[3]
            }
        })

    return {"results": results}

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)

    generated_ids = model_blip.generate(**inputs)
    description = processor.decode(generated_ids[0], skip_special_tokens=True)

    return {"description": description}
