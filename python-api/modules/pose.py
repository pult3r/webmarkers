from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
def detect_pose(path): return model(path)[0].tojson()