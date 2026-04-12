import json
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

from lib.semantic_search import cosine_similarity
from consts import DOCS_JSON_PATH, CLIP_TEXT_EMBEDDINGS_SAVE_PATH, CLIP_MODEL, DEFAULT_DESCRIPTION_LEN, DEFAULT_TOP_K


def verify_image_embedding(image_path: str, model: str=CLIP_MODEL):
    ms = MultimodalSearch(model)
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def clip_search_with_image(image_path: str, limit: int=DEFAULT_TOP_K, model: str=CLIP_MODEL, docs_json_path: str=DOCS_JSON_PATH):
    with open(docs_json_path, 'r') as f:
        docs_data = json.load(f)["movies"]

    ms = MultimodalSearch(model)
    ms.load_or_create_embeddings(docs_data)
    search_results = ms.search_with_image(image_path, limit)
    for i, doc in enumerate(search_results):
        print(f"{i + 1}. {doc['title']} (similarity: {doc['score']:.3f})")
        print(f"   {doc['description'][:DEFAULT_DESCRIPTION_LEN]}\n")


class MultimodalSearch:
    def __init__(self, model: str=CLIP_MODEL):
        self.model = SentenceTransformer(model)
        self.text_embeddings = {}
        self.documents = []
        self.document_map = {}
    
    def embed_image(self, image_path: str) -> list:
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]
        return embedding
    
    def search_with_image(self, image_path: str, limit: int=DEFAULT_TOP_K) -> list[dict]:
        image_emb = self.embed_image(image_path)
        similarities = {}
        for doc_id, text_emb in self.text_embeddings.items():
            similarity = cosine_similarity(text_emb, image_emb)
            similarities[doc_id] = similarity
        
        most_relevant_docs = sorted(list(similarities.items()), key=lambda it: it[1], reverse=True)[:limit]
        format_result = []
        for doc_id, score in most_relevant_docs:
            format_result.append({
                "id": doc_id,
                "title": self.document_map[doc_id]["title"],
                "description": self.document_map[doc_id]["description"],
                "score": round(score, 6),
            })
            
        return format_result
    
    def build_embeddings(self, documents: list[dict], emb_save_path: str=CLIP_TEXT_EMBEDDINGS_SAVE_PATH) -> dict[int, list]:
        self.documents = documents
        for doc in tqdm(documents):
            self.document_map[doc["id"]] = doc
            doc_repr = f"{doc['title']}: {doc['description']}"
            doc_emb = self.model.encode([doc_repr])[0]
            self.text_embeddings[doc["id"]] = doc_emb
    
        self.save(emb_save_path)
        return self.text_embeddings
    
    def save(self, emb_save_path: str=CLIP_TEXT_EMBEDDINGS_SAVE_PATH):
        if os.path.exists(emb_save_path):
            raise ValueError(f"CLIP text embeddings aleardy exists: {emb_save_path}")
        
        np.save(emb_save_path, self.text_embeddings)
        print("Saved MultimodalSearch.text_embeddings successfully")

    def load_or_create_embeddings(self, documents: list[dict], emb_save_path: str=CLIP_TEXT_EMBEDDINGS_SAVE_PATH) -> dict[int, list]:
        if not os.path.exists(emb_save_path):
            return self.build_embeddings(documents, emb_save_path)

        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        self.text_embeddings = np.load(emb_save_path, allow_pickle=True).item()
        return self.text_embeddings
