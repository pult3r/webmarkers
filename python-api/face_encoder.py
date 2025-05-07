import face_recognition
import os
import pickle

KNOWN_FACES_DIR = "faces"
ENCODINGS_FILE = "face_encodings.pkl"

known_faces = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_faces.append({
                "name": os.path.splitext(filename)[0],
                "encoding": encodings[0]
            })

# Zapis do pliku
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(known_faces, f)

print(f"Zapisano {len(known_faces)} twarzy do {ENCODINGS_FILE}")
