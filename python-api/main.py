from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8n.pt")  # lub inny model YOLO

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(image)
    return {"detections": results[0].tojson()}

@app.post("/face-detect")
async def face_detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_img)

    # Zwraca listę wykrytych twarzy w formacie (top, right, bottom, left)
    faces = [
        {"top": top, "right": right, "bottom": bottom, "left": left}
        for top, right, bottom, left in face_locations
    ]

    return {"faces": faces}
