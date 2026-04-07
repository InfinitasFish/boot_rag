import os
import json

from lib.inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from consts import DOCS_JSON_PATH, INDEX_DB_PATH, DEFAULT_TOP_K, DEFAULT_ALPHA_WEIGHT


def minmax_normalize_scores(scores: list[float]) -> list[float]:
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    for i in range(len(scores)):
        scores[i] = (scores[i] - min_score) / (max_score - min_score)
    
    return scores


def hybrid_score_search(query: str, alpha: float=DEFAULT_ALPHA_WEIGHT, limit: int=DEFAULT_TOP_K, docs_json_path: str=DOCS_JSON_PATH):
    with open(docs_json_path, 'r') as f:
        docs_data = json.load(f)["movies"]
    hs = HybridSearch(docs_data)
    search_results = hs.weighted_search(query, alpha, limit)
    for i, doc in enumerate(search_results):
        print(f"{i + 1}. {doc['title']}\n   Hybrid Score: {doc['']}")

# 1. Paddington
#   Hybrid Score: 1.000
#   BM25: 1.000, Semantic: 1.000
#   Deep in the rainforests of Peru, a young bear lives peacefully with his Aunt Lucy and Uncle Pastuzo,...
# 2. The Indian in the Cupboard
#   Hybrid Score: 0.943
#   BM25: 0.966, Semantic: 0.850
#   On his ninth birthday, Omri receives an old cupboard from his brother Gillon (Vincent Kartheiser) an...
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
        return norm_scores

    def weighted_search(self, query: str, alpha: float=DEFAULT_ALPHA_WEIGHT, limit: int=DEFAULT_TOP_K) -> list[dict]:
        # who's 500
        bm25_search_results = self._bm25_search(query, limit * 500)
        semantic_search_results = self._semantic_chunk_search(query, litmit * 500)

        bm25_search_results_norm = self.norm_search_scores(bm25_search_results)
        semantic_search_results_norm = self.norm_search_scores(semantic_search_results)

        relevant_joined_docs_idxs = set([doc["id"] for doc in bm25_search_results_norm] + [doc["id"] for doc in semantic_search_results_norm])
        hybrid_scores_docs = {}
        for doc_id in relevant_joined_docs_ids:
            hybrid_score = bm25_search_results_norm[doc_id] * alpha + semantic_search_results_norm[doc_id] * (1 - alpha)
            hybrid_scores_docs[doc_id] = hybrid_score
        
        most_relevant_docs = sorted(list(hybrid_scores_docs.items()), key=lambda it: it[1], reverse=True)[:limit]
        search_results = []

        for doc_id, hyb_score in most_relevant_docs:
            search_results.append({
                "title": self.semantic_search.document_map[doc_id]["title"],
                "bm25score": bm25_search_results_norm[doc_id],
                "semantic_score": semantic_search_results_norm[doc_id],
                "hybrid_score": hyb_score,
                "description": self.semantic_search.document_map[doc_id]["descrption"][:100],
            })

        return search_results

    def rrf_search(self, query: str, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

