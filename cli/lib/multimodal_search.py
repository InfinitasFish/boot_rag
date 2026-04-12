from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

from consts import CLIP_MODEL


def verify_image_embedding(image_path: str, model: str=CLIP_MODEL):
    ms = MultimodalSearch(model)
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


class MultimodalSearch:
    def __init__(self, model: str=CLIP_MODEL):
        self.model = SentenceTransformer(model)
    
    def embed_image(self, image_path: str) -> list:
        image = Image.open(image_path)
        print("start")
        embedding = self.model.encode([image])[0]
        print("end")
        return embedding
