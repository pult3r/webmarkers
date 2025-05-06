from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8n.pt")  # Możesz zastąpić YOLO-NAS

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(image)
    return {"detections": results[0].tojson()}