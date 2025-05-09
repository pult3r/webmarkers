from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import face_recognition
import pickle
from ultralytics import YOLO
from PIL import Image
import io
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Wczytaj znane twarze
with open("face_encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Inicjalizacja FastAPI
app = FastAPI()

# Inicjalizacja YOLO
yolo_model = YOLO("yolov8n.pt")

# Inicjalizacja BLIP-2
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model_vqa = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

# --- ENDPOINTY ---

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = yolo_model.predict(image)
    return {"detections": results[0].tojson()}

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

@app.post("/ask-image-question")
async def ask_image_question(
    file: UploadFile = File(...), 
    question: str = "What is happening in the image?",
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # BLIP-2: odpowiedź po angielsku
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    out = model_vqa.generate(**inputs)
    answer_en = processor.decode(out[0], skip_special_tokens=True)

    return {
        "question": question,
        "answer": answer_en
    }
