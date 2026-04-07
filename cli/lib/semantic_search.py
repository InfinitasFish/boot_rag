import json
import os
import re
from math import ceil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

from consts import (DOCS_JSON_PATH, TEXT_EMBEDDING_MODEL, EMBEDDINGS_SAVE_PATH, DEFAULT_TOP_K, DEFAULT_CHUNK_SIZE, DEFAULT_SEMANTIC_CHUNK_SIZE, 
                    DEFAULT_OVERLAP_SIZE, DEFAULT_SEARCH_OVERLAP_SIZE, CHUNK_EMBEDDINGS_SAVE_PATH, CHUNK_META_SAVE_PATH)


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


def semantic_search(query: str, limit: int=DEFAULT_TOP_K, model: str=TEXT_EMBEDDING_MODEL, docs_json_path: str=DOCS_JSON_PATH):
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
    if overlap_size >= chunk_size:
        raise ValueError("Kys")
    text_words = text.split()
    text_chunks = [' '.join(text_words[max(0, i * (chunk_size - overlap_size)): (chunk_size if i == 0 else (i + 1) * (chunk_size - overlap_size) + overlap_size)]) for i in range(ceil((len(text_words) - chunk_size) / (chunk_size - overlap_size)) + 1)]
    return text_chunks


def split_text_chunks_semantic(text: str, chunk_size: int=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap_size: int=DEFAULT_OVERLAP_SIZE) -> list:
    if overlap_size >= chunk_size:
        raise ValueError("Kys")

    text = text.strip()
    if not text:
        return []

    text_sentences = re.split(r"(?<=[.!?])\s+", text)
    text_chunks = [' '.join(text_sentences[max(0, i * (chunk_size - overlap_size)): (chunk_size if i == 0 else (i + 1) * (chunk_size - overlap_size) + overlap_size)]) for i in range(ceil((len(text_sentences) - chunk_size) / (chunk_size - overlap_size)) + 1)]
    return text_chunks


def semantic_chunk_search(query: str, limit: int=DEFAULT_TOP_K, model: str=TEXT_EMBEDDING_MODEL, docs_json_path: str=DOCS_JSON_PATH):
    ss = ChunkedSemanticSearch(model=model)
    with open(docs_json_path, 'r') as f:
        docs_data = json.load(f)["movies"]
    ss.load_or_create_chunk_embeddings(docs_data)

    relevant_docs_format = ss.search(query, limit)
    for i, doc in enumerate(relevant_docs_format):
        print(f"\n{i + 1}. {doc['title']} (score: {doc['score']:.4f})")
        print(f"   {doc['document']}...")


def build_chunks_embed(model: str=TEXT_EMBEDDING_MODEL, docs_path: str=DOCS_JSON_PATH, chunk_size: int=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap_size: int=DEFAULT_SEARCH_OVERLAP_SIZE):
    ss = ChunkedSemanticSearch(model=model)
    with open(docs_path, 'r') as f:
        docs_data = json.load(f)["movies"]
    ss.load_or_create_chunk_embeddings(docs_data, chunk_size, overlap_size)
    print(f"Generated {len(ss.chunk_metadata)} chunked embeddings")


class SemanticSearch:
    def __init__(self, model: str=TEXT_EMBEDDING_MODEL):
        self.model = SentenceTransformer(model)
        self.embeddings = None
        self.documents = None
        self.document_map = None

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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model: str=TEXT_EMBEDDING_MODEL):
        super().__init__(model)
        self.chunk_text = None
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def search(self, query: str, limit: int=DEFAULT_TOP_K) -> list[dict]:
        if not self.chunk_embeddings or not self.chunk_metadata:
            raise ValueError("Embeddings aren't built/loaded, can't search")
        
        docs_agg_scores = {}
        query_emb = self.generate_embedding(query)
        for doc_id, chunk_embeds in tqdm(self.chunk_embeddings.items()):
            # calculating max similarity as document score
            doc_score = -float("inf")
            # kinda slow I guess
            for chunk in chunk_embeds:
                similarity = cosine_similarity(query_emb, chunk)
                doc_score = similarity if similarity > doc_score else doc_score
    
            docs_agg_scores[doc_id] = doc_score
        
        most_relevant_docs = sorted(list(docs_agg_scores.items()), key=lambda it: it[1], reverse=True)[:limit]
        format_result = []
        for doc_id, score in most_relevant_docs:
            format_result.append({
                "id": doc_id,
                "title": self.document_map[doc_id]["title"],
                "document": self.document_map[doc_id]["description"][:100],
                "score": round(score, 6),
            })
            
        return format_result


    # define new build_embeddings for chunks
    def build_chunk_embeddings(self, documents: list[dict[int, str]], chunk_size: int=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap_size: int=DEFAULT_SEARCH_OVERLAP_SIZE,
                               emb_save_path: str=CHUNK_EMBEDDINGS_SAVE_PATH, meta_save_path: str=CHUNK_META_SAVE_PATH) -> dict[int, list]:
        self.documents = documents
        self.chunk_text = {}
        self.chunk_embeddings = {}
        # basically I've changed the supposed implementation a bit, so metadata is redundant
        self.chunk_metadata = []
        for doc in tqdm(documents):
            self.document_map[doc["id"]] = doc
            if not doc["description"].strip():
                continue

            doc_chunks = split_text_chunks_semantic(doc["description"], chunk_size, overlap_size)
            self.chunk_text[doc["id"]] = doc_chunks

            chunk_embeddings = self.model.encode(doc_chunks)
            self.chunk_embeddings[doc["id"]] = chunk_embeddings

            for i, chunk in enumerate(chunk_embeddings):
                self.chunk_metadata.append({"movie_idx": doc["id"], "chunk_idx": i, "total_chunks": len(chunk_embeddings)})

        self.save(emb_save_path, meta_save_path)
        return self.chunk_embeddings

    # redefine save method
    def save(self, emb_save_path: str=CHUNK_EMBEDDINGS_SAVE_PATH, meta_save_path: str=CHUNK_META_SAVE_PATH):
        if os.path.exists(emb_save_path):
            raise ValueError(f"Chunk embeddings aleardy exists: {emb_save_path}")
        if os.path.exists(meta_save_path):
            raise ValueError(f"Chunk meta data aleardy exists: {meta_save_path}")

        np.save(emb_save_path, self.chunk_embeddings)
        with open(CHUNK_META_SAVE_PATH, 'w') as f:
            json.dump(self.chunk_metadata, f)
        print("Saved ChunkedSemanticSearch.chunk_embeddings and .chunk_metadata successfully")
    
    # define new load_or_create_embeddings
    def load_or_create_chunk_embeddings(self, documents: list[dict], chunk_size: int=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap_size: int=DEFAULT_SEARCH_OVERLAP_SIZE,
                                        emb_save_path: str=CHUNK_EMBEDDINGS_SAVE_PATH, meta_save_path: str=CHUNK_META_SAVE_PATH) -> dict[int, list]:
        # cache doesn't exists -> build embeddings and return
        if not os.path.exists(emb_save_path) and not os.path.exists(meta_save_path):
            return self.build_chunk_embeddings(documents, chunk_size, overlap_size, emb_save_path, meta_save_path)
        
        # cache exists -> populate self.documents, .documents_map and .chunk_text, load embeddings and meta data, return
        self.documents = documents
        self.chunk_text = {}
        for doc in tqdm(documents):
            self.document_map[doc["id"]] = doc
            if not doc["description"].strip():
                continue

            # maybe it's better to save it too or populate using loaded meta data, but I'm not sure
            doc_chunks = split_text_chunks_semantic(doc["description"], chunk_size, overlap_size)
            self.chunk_text[doc["id"]] = doc_chunks
        
        self.chunk_embeddings = np.load(emb_save_path, allow_pickle=True).item()
        with open(meta_save_path, 'r') as f:
            self.chunk_metadata = json.load(f)
        return self.chunk_embeddings
        
