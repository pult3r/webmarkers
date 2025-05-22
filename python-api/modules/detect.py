from ultralytics import YOLO
model = YOLO('yolov8n.pt')
def detect_objects(path): return model(path)[0].tojson()