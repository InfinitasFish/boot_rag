import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

from consts import DOCS_JSON_PATH, TEXT_EMBEDDING_MODEL, EMBEDDINGS_SAVE_PATH


def verify_model(model: str=TEXT_EMBEDDING_MODEL):
    # dont streSS
    ss = SemanticSearch(model=model)
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def verify_embeddings(model: str=TEXT_EMBEDDING_MODEL, docs_json_path: str=DOCS_JSON_PATH):
    ss = SemanticSearch(model=model)
    with open(docs_json_path, 'r') as f:
        docs_data = json.load(f)["movies"]
    embeddings = ss.load_or_create_embeddings(docs_data)
    num_emb = len(embeddings)
    emb_dim = np.array(list(embeddings.values())[0]).shape[0]
    print(f"Number of docs:   {len(docs_data)}")
    print(f"Embeddings shape: {num_emb} vectors in {emb_dim} dimensions")

def embed_text(text: str, model: str=TEXT_EMBEDDING_MODEL):
    ss = SemanticSearch(model=model)
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


class SemanticSearch:
    def __init__(self, model: str=TEXT_EMBEDDING_MODEL):
        self.model = SentenceTransformer(model)
        self.embeddings = {}
        self.documents = {}
        self.document_map = {}

    def generate_embedding(self, text: str) -> list:
        if not text.strip():
            raise ValueError("Empty string in SemanticSearch.generate_embedding()")
        embedding = self.model.encode([text])[0]
        return embedding
    
    def build_embeddings(self, documents: list[dict[int, str]], emb_save_path: str=EMBEDDINGS_SAVE_PATH) -> dict[int, list]:
        self.documents = documents
        for doc in tqdm(documents):
            self.document_map[doc["id"]] = doc
            doc_repr = f"{doc['title']}: {doc['description']}"
            doc_emb = self.model.encode([doc_repr])[0]
            self.embeddings[doc["id"]] = doc_emb
        
        self.save(emb_save_path)
        return self.embeddings
    
    def save(self, emb_save_path: str=EMBEDDINGS_SAVE_PATH):
        emb_dir_save_path = os.path.dirname(os.path.abspath(emb_save_path))
        if not os.path.exists(emb_dir_save_path):
            os.mkdir(emb_dir_save_path)
        if os.path.exists(emb_save_path):
            raise ValueError(f"{emb_save_path} already exists, delete manually before saving db")

        np.save(emb_save_path, self.embeddings)
        print("Saved SemanticSearch.embeddings successfully")
    
    def load_or_create_embeddings(self, documents: list[dict[int, str]], emb_save_path: str=EMBEDDINGS_SAVE_PATH) -> dict[int, list]:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if os.path.exists(emb_save_path):
            self.embeddings = np.load(emb_save_path, allow_pickle=True).item()
            if len(self.embeddings) != len(self.documents):
                raise ValueError(f"Loaded embeddings from {emb_save_path} have different size compared to documents")
            else:
                return self.embeddings
        else:
            return self.build_embeddings(documents, emb_save_path)


