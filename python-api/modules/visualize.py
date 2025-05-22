import base64
from modules.clip_module import map

def visualize_image_by_id(id):
    return {'path': map.get(id)}
def visualize_base64_by_id(id):
    with open(map.get(id), 'rb') as f:
        return {'image_base64': base64.b64encode(f.read()).decode()}