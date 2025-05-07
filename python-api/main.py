from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import face_recognition
import pickle
from ultralytics import YOLO


#from fastapi import HTTPException

# Ładowanie znanych twarzy
with open("face_encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)

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
