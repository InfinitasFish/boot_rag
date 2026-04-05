import json
import os
from math import ceil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

from consts import DOCS_JSON_PATH, TEXT_EMBEDDING_MODEL, EMBEDDINGS_SAVE_PATH, DEFAULT_TOP_K, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_SIZE


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


def embed_query_text(query: str, model: str=TEXT_EMBEDDING_MODEL):
    ss = SemanticSearch(model=model)
    query_emb = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {query_emb[:3]}")
    print(f"Shape: {query_emb.shape}")


def embed_text(text: str, model: str=TEXT_EMBEDDING_MODEL):
    ss = SemanticSearch(model=model)
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def search(query: str, limit: int=DEFAULT_TOP_K, model: str=TEXT_EMBEDDING_MODEL, docs_json_path: str=DOCS_JSON_PATH):
    ss = SemanticSearch(model=model)

    with open(docs_json_path, 'r') as f:
        docs_data = json.load(f)["movies"]
    ss.load_or_create_embeddings(docs_data)

    top_matches_repr = ss.search(query, limit)
    for i, doc in enumerate(top_matches_repr):
        print(f"{i+1}. {doc['title']} (score: {doc['score']:.4f})\n  {doc['description'][:150]} ...\n")


# list but ndarray but who tf cares
def cosine_similarity(vec1: list, vec2: list) -> float:
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 * norm2 == 0.0:
        return 0.0
    
    return dot / (norm1 * norm2)


def split_text_chunks(text: str, chunk_size: int=DEFAULT_CHUNK_SIZE, overlap_size: int=DEFAULT_OVERLAP_SIZE) -> list:
    text_words = text.split()
    text_chunks = [' '.join(text_words[max(0, i * (chunk_size - overlap_size)): (chunk_size if i == 0 else (i + 1) * (chunk_size - overlap_size) + overlap_size)]) for i in range(ceil((len(text_words) - chunk_size) / (chunk_size - overlap_size)) + 1)]
    return text_chunks


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
    
    def search(self, query: str, limit: int=DEFAULT_TOP_K) -> list[dict]:
        if not self.embeddings:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first")
        
        query_emb = self.generate_embedding(query)
        similarities = {}
        for doc_id, doc_emb in self.embeddings.items():
            similarity = cosine_similarity(query_emb, doc_emb)
            similarities[doc_id] = similarity
        
        top_matches = sorted(list(similarities.items()), key=lambda it: it[1], reverse=True)[:limit]
        top_matches_repr = []
        for doc_id, similarity in top_matches:
            top_matches_repr.append({
                "score": similarity, 
                "title": self.document_map[doc_id]["title"],
                "description": self.document_map[doc_id]["description"],
                })
        
        return top_matches_repr
    
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

