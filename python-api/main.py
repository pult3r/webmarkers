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
import os
import faiss
import clip
from fastapi.responses import FileResponse, JSONResponse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import base64

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








# Inicjalizacja CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

# Inicjalizacja FAISS
index = None
index_path = "clip_embeddings.index"
metadata = []

def init_faiss_index(dim=512):
    global index
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(dim)


init_faiss_index()


@app.post("/clip-embed")
async def clip_embed(file: UploadFile = File(...)):
    # Wczytanie obrazu
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    image_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    cv2.imwrite(image_path, img)


    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    # Detekcja obiektów YOLO
    results = model.predict(rgb_img)
    
    # Przygotowanie wyników
    detections = []
    embeddings_list = []
    
    for box in results[0].boxes:
        # Współrzędne bounding boxa
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Wycięcie ROI
        roi = pil_img.crop((x1, y1, x2, y2))
        
        # Generowanie embeddingu CLIP
        roi_processed = preprocess_clip(roi).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model_clip.encode_image(roi_processed).cpu().numpy()
        
        # Dodanie do FAISS
        faiss_id = len(metadata)
        index.add(embedding.astype('float32'))
        metadata.append({
            "faiss_id": faiss_id,
            "box": [x1, y1, x2, y2],
            "class_id": int(box.cls),
            "conf": float(box.conf),
            "image_path": image_path  # Kluczowa zmiana!
        })
        
        detections.append({
            "box": [x1, y1, x2, y2],
            "class_id": int(box.cls),
            "confidence": float(box.conf),
            "embedding": embedding.tolist()[0],
            "faiss_id": faiss_id
        })
    
    # Zapisanie indeksu FAISS
    faiss.write_index(index, index_path)
    
    return {
        "detections": detections,
        "message": f"Added {len(detections)} new embeddings to FAISS index"
    }

@app.post("/clip-search")
async def clip_search(file: UploadFile = File(...), k: int = 5):
    # Wczytanie obrazu zapytania
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    
    # Przetworzenie całego obrazu przez CLIP
    img_processed = preprocess_clip(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model_clip.encode_image(img_processed).cpu().numpy()
    
    # Wyszukiwanie w FAISS
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    # Przygotowanie wyników
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx >= 0 and idx < len(metadata):  # Sprawdzenie poprawności indeksu
            meta = metadata[idx]
            results.append({
                "rank": i+1,
                "distance": float(distance),
                "faiss_id": int(idx),
                "box": meta["box"],
                "class_id": meta["class_id"],
                "confidence": meta["conf"]
            })
    
    return {"results": results}

@app.get("/clip-metadata/{faiss_id}")
async def get_clip_metadata(faiss_id: int):
    if faiss_id < 0 or faiss_id >= len(metadata):
        raise HTTPException(status_code=404, detail="ID not found")
    return metadata[faiss_id]


@app.post("/text-search")
async def text_search(query: str = "sport bike", k: int = 5):
    # Generowanie embeddingu tekstowego za pomocą CLIP
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = model_clip.encode_text(text_input).cpu().numpy()
    
    # Wyszukiwanie w FAISS
    distances, indices = index.search(text_embedding.astype('float32'), k)
    
    # Przygotowanie wyników
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx >= 0 and idx < len(metadata):
            item = metadata[idx]
            results.append({
                "rank": i+1,
                "distance": float(distance),
                "faiss_id": int(idx),
                "box": item["box"],
                "class_id": item["class_id"],
                "confidence": item.get("conf", 0),
                "class_name": model.names[int(item["class_id"])]  # Nazwa klasy z YOLO
            })
    
    return {"query": query, "results": results}


@app.get("/visualize/{faiss_id}")
async def visualize(faiss_id: int):
    """
    Wizualizuje obiekt o podanym faiss_id z bounding boxem na obrazie.
    Wymaga przechowywania ścieżek do obrazów w metadata.
    """
    if faiss_id < 0 or faiss_id >= len(metadata):
        raise HTTPException(status_code=404, detail="ID not found")
    
    item = metadata[faiss_id]
    
    # Sprawdź, czy obraz istnieje w metadata
    if "image_path" not in item:
        raise HTTPException(
            status_code=400,
            detail="Image path not found in metadata. Add 'image_path' when saving embeddings."
        )
    
    # Wczytaj obraz
    if not os.path.exists(item["image_path"]):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    image = cv2.imread(item["image_path"])
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Stwórz figurę matplotlib
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(rgb_img)
    
    # Narysuj bounding box
    box = item["box"]
    rect = patches.Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=3, edgecolor="#FF0000", facecolor="none"
    )
    ax.add_patch(rect)
    
    # Dodaj opis
    class_name = model.names[int(item["class_id"])]
    plt.title(f"{class_name} (ID: {faiss_id}, Conf: {item.get('conf', 0):.2f})", fontsize=14)
    plt.axis("off")
    
    # Zapisz tymczasowo
    output_path = f"temp_visualization_{faiss_id}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=100, pad_inches=0.1)
    plt.close()
    
    return FileResponse(output_path)


@app.get("/visualize-base64/{faiss_id}")
async def visualize_base64(faiss_id: int):
    """
    Wizualizuje obiekt o podanym faiss_id i zwraca obraz w formacie base64.
    
    Parametry:
    - faiss_id: ID obiektu w metadata (pobrane z wyników /text-search)
    
    Zwraca:
    - JSON z obrazem w base64 i metadanymi
    """
    
    # 1. Sprawdź czy faiss_id istnieje w metadata
    if faiss_id < 0 or faiss_id >= len(metadata):
        raise HTTPException(
            status_code=404,
            detail=f"faiss_id {faiss_id} not found. Max available ID: {len(metadata)-1}"
        )
    
    item = metadata[faiss_id]
    
    # 2. Sprawdź czy ścieżka do obrazu istnieje
    if "image_path" not in item or not os.path.exists(item["image_path"]):
        raise HTTPException(
            status_code=400,
            detail="Image path not found in metadata or file missing"
        )
    
    try:
        # 3. Wczytaj obraz
        image = cv2.imread(item["image_path"])
        if image is None:
            raise HTTPException(status_code=500, detail="Failed to read image file")
            
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 4. Przygotuj wykres
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(rgb_img)
        
        # 5. Narysuj bounding box
        box = item["box"]
        rect = patches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1],
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 6. Dodaj opis
        class_name = model.names[int(item["class_id"])] if hasattr(model, 'names') else f"class_{item['class_id']}"
        plt.title(f"{class_name} (ID: {faiss_id}, Conf: {item.get('conf', 0):.2f})")
        plt.axis('off')
        
        # 7. Zapisz do bufora w pamięci (bez pliku tymczasowego)
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        
        # 8. Konwertuj do base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # 9. Przygotuj odpowiedź
        return JSONResponse(
            content={
                "status": "success",
                "faiss_id": faiss_id,
                "class_id": item["class_id"],
                "class_name": class_name,
                "confidence": float(item.get("conf", 0)),
                "box": item["box"],
                "image_base64": f"data:image/png;base64,{img_base64}"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Visualization failed: {str(e)}"
        )