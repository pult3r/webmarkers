import cv2
import numpy as np
from sklearn.cluster import KMeans
def detect_dominant_colors(path, k=3):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    return {'dominant_colors': kmeans.cluster_centers_.astype(int).tolist()}