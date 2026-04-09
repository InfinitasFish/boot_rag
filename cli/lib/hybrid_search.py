import os
import json

from lib.inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from consts import DOCS_JSON_PATH, INDEX_DB_PATH, DEFAULT_DESCRIPTION_LEN, DEFAULT_TOP_K, DEFAULT_ALPHA_WEIGHT, DEFAULT_RRF_K


def minmax_normalize_scores(scores: list[float]) -> list[float]:
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    for i in range(len(scores)):
        scores[i] = (scores[i] - min_score) / (max_score - min_score)
    
    return scores


def hybrid_norm_score_search(query: str, alpha: float=DEFAULT_ALPHA_WEIGHT, limit: int=DEFAULT_TOP_K, docs_json_path: str=DOCS_JSON_PATH):
    with open(docs_json_path, 'r') as f:
        docs_data = json.load(f)["movies"]
    hs = HybridSearch(docs_data)
    search_results = hs.weighted_search(query, alpha, limit)
    for i, doc in enumerate(search_results):
        print(f"{i + 1}. {doc['title']}\n   Hybrid Score: {doc['hybrid_score']:.3f}")
        print(f"   BM25: {doc['bm25score']:.3f}, Semantic: {doc['semantic_score']:.3f}")
        print(f"   {doc['description'][:DEFAULT_DESCRIPTION_LEN]}...")


def hybrid_rrf_score_search(query: str, k: float=DEFAULT_RRF_K, limit: int=DEFAULT_TOP_K, docs_json_path: str=DOCS_JSON_PATH):
    with open(docs_json_path, 'r') as f:
        docs_data = json.load(f)["movies"]
    hs = HybridSearch(docs_data)
    search_results = hs.rrf_search(query, k, limit)
    for i, doc in enumerate(search_results):
        print(f"{i + 1}. {doc['title']}\n   RRF Score: {doc['rrf_score']:.3f}")
        print(f"   BM25 Rank: {doc['bm25_rank']}, Semantic Rank: {doc['semantic_rank']}")
        print(f"   {doc['description'][:DEFAULT_DESCRIPTION_LEN]}...")


class HybridSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_DB_PATH):
            self.idx.build()
            self.idx.save()
        else:
            self.idx.load()
        
    def _bm25_search(self, query: str, limit: int=DEFAULT_TOP_K) -> list[dict]:
        return self.idx.bm25_search(query, limit)
    
    def _semantic_chunk_search(self, query: str, limit: int=DEFAULT_TOP_K) -> list[dict]:
        return self.semantic_search.search(query, limit)

    def norm_search_scores(self, search_results: list[dict]) -> dict[int, float]:
        scores = [doc["score"] for doc in search_results]
        norm_scores = minmax_normalize_scores(scores)
        norm_docs = {}
        for i, doc in enumerate(search_results):
            norm_docs[doc["id"]] = norm_scores[i]
        return norm_docs

    def rrf_scores(self, search_results: list[dict], k: float=DEFAULT_RRF_K) -> dict[int, float]:
        rrf_docs = {}
        for i, doc in enumerate(search_results):
            rrf_docs[doc["id"]] = 1 / (k + i + 1)
        return rrf_docs

    def weighted_search(self, query: str, alpha: float=DEFAULT_ALPHA_WEIGHT, limit: int=DEFAULT_TOP_K) -> list[dict]:
        # who's 500
        bm25_search_results = self._bm25_search(query, limit * 500)
        semantic_search_results = self._semantic_chunk_search(query, limit * 500)

        bm25_search_results_norm = self.norm_search_scores(bm25_search_results)
        semantic_search_results_norm = self.norm_search_scores(semantic_search_results)

        # collecting found idxs by groups and calculating score accordingly 
        bm25_only_group = set(list(bm25_search_results_norm.keys())) - set(list(semantic_search_results_norm.keys()))
        semantic_only_group = set(list(semantic_search_results_norm.keys())) - set(list(bm25_search_results_norm.keys()))
        joined_group = set(list(bm25_search_results_norm.keys())) & set(list(semantic_search_results_norm.keys()))
        hybrid_scores_docs = {}

        for doc_id in bm25_only_group:
            hybrid_score = bm25_search_results_norm[doc_id] * alpha
            hybrid_scores_docs[doc_id] = hybrid_score
        
        for doc_id in semantic_only_group:
            hybrid_score = semantic_search_results_norm[doc_id] * (1 - alpha)
            hybrid_scores_docs[doc_id] = hybrid_score

        for doc_id in joined_group:
            hybrid_score = bm25_search_results_norm[doc_id] * alpha + semantic_search_results_norm[doc_id] * (1 - alpha)
            hybrid_scores_docs[doc_id] = hybrid_score
        
        most_relevant_docs = sorted(list(hybrid_scores_docs.items()), key=lambda it: it[1], reverse=True)[:limit]
        search_results = []

        for doc_id, hyb_score in most_relevant_docs:
            search_results.append({
                "id": doc_id,
                "title": self.semantic_search.document_map[doc_id]["title"],
                "bm25score": bm25_search_results_norm[doc_id],
                "semantic_score": semantic_search_results_norm[doc_id],
                "hybrid_score": hyb_score,
                "description": self.semantic_search.document_map[doc_id]["description"],
            })

        return search_results

    def rrf_search(self, query: str, k: float=DEFAULT_RRF_K, limit: int=DEFAULT_TOP_K) -> list[dict]:
        bm25_search_results = self._bm25_search(query, limit * 500)
        semantic_search_results = self._semantic_chunk_search(query, limit * 500)

        bm25_search_results_rrf = self.rrf_scores(bm25_search_results, k)
        bm25_search_results_ranks = {doc["id"]: i + 1 for i, doc in enumerate(bm25_search_results)}
        semantic_search_results_rrf = self.rrf_scores(semantic_search_results, k)
        semantic_search_results_ranks = {doc["id"]: i + 1 for i, doc in enumerate(semantic_search_results)}

        # collecting found idxs by groups and calculating score accordingly
        bm25_only_group = set(list(bm25_search_results_rrf.keys())) - set(list(semantic_search_results_rrf.keys()))
        semantic_only_group = set(list(semantic_search_results_rrf.keys())) - set(list(bm25_search_results_rrf.keys()))
        joined_group = set(list(bm25_search_results_rrf.keys())) & set(list(semantic_search_results_rrf.keys()))
        
        hybrid_scores_docs = {}
        for doc_id in bm25_only_group:
            hybrid_scores_docs[doc_id] = bm25_search_results_rrf[doc_id]
        for doc_id in semantic_only_group:
            hybrid_scores_docs[doc_id] = semantic_search_results_rrf[doc_id]
        for doc_id in joined_group:
            hybrid_scores_docs[doc_id] = bm25_search_results_rrf[doc_id] + semantic_search_results_rrf[doc_id]

        most_relevant_docs = sorted(list(hybrid_scores_docs.items()), key=lambda it: it[1], reverse=True)[:limit]
        search_results = []
        for doc_id, hyb_score in most_relevant_docs:
            search_results.append({
                "id": doc_id,
                "title": self.semantic_search.document_map[doc_id]["title"],
                "bm25_rank": bm25_search_results_ranks[doc_id],
                "semantic_rank": semantic_search_results_ranks[doc_id],
                "rrf_score": hyb_score,
                "description": self.semantic_search.document_map[doc_id]["description"],
            })

        return search_results
