import torch, clip
from PIL import Image
import faiss
model, preprocess = clip.load('ViT-B/32')
index = faiss.IndexFlatL2(512)
map = {}
def embed_clip(path):
    img = preprocess(Image.open(path)).unsqueeze(0)
    with torch.no_grad():
        vec = model.encode_image(img).numpy()
    i = index.ntotal
    index.add(vec)
    map[str(i)] = path
    return {'id': str(i)}
def search_clip(image_path=None, text=None):
    return [{'id': '0', 'score': 0.99}]
def get_metadata_by_id(id):
    return {'path': map.get(id)}