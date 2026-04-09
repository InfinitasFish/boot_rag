import json
import pickle
import os
from collections import Counter
from math import log
# import sys
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from consts import DOCS_JSON_PATH, INDEX_DB_PATH, DOCMAP_PATH, TERM_FREQ_PATH, DOC_LENGTHS_PATH, BM25_K1, BM25_B, DEFAULT_TOP_K
from lib.preprocess import preprocess_text_to_tokens_pipe


# forward index maps location to value,
# an inverted index maps value to location
# e.g. token 'matrix' -> [1, 5, 10] movies ids
class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        text_tokens = preprocess_text_to_tokens_pipe(text)
        self.doc_lengths[doc_id] = len(text_tokens)
        self.term_frequencies[doc_id] = Counter(text_tokens)
        for token in self.term_frequencies[doc_id]:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = set([doc_id])
    
    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def get_documents(self, term):
        term = term.lower()
        idx = sorted(self.index[term])
        return idx
    
    def get_tf(self, doc_id, term):
        if not doc_id in self.docmap:
            raise ValueError(f"Unknown doc_id in InvertedIndex.get_tf(): {doc_id}")
        term_token = preprocess_text_to_tokens_pipe(term)
        if len(term_token) != 1:
            raise ValueError("InvertedIndex.get_tf() accepts only one term\n")
        return self.term_frequencies[doc_id][term_token[0]]
    
    def get_idf(self, term):
        term_token = preprocess_text_to_tokens_pipe(term)
        if len(term_token) != 1:
            raise ValueError("InvertedIndex.get_idf() accepts only one term\n")

        # avoid zero-division
        doc_freq = len(self.index.get(term_token[0], []))
        term_idf_score = log((len(self.docmap) + 1) / (doc_freq + 1))
        return term_idf_score
    
    def get_tf_idf(self, doc_id, term):
        return self.get_tf(doc_id, term) * self.get_idf(term)
    
    # fixes original term-freq saturation by applying diminishing returns
    def get_bm25_tf(self, doc_id: int, term: str, k1: float=BM25_K1, b: float=BM25_B) -> float:
        if not doc_id in self.docmap:
            raise ValueError(f"Unknown doc_id in InvertedIndex.get_bm25_tf(): {doc_id}")

        term_token = preprocess_text_to_tokens_pipe(term)
        if len(term_token) != 1:
            raise ValueError("InvertedIndex.get_bm25_tf() accepts only one term\n")
        
        # BM25 adjusts term frequency based on document length (penalize long docs, boost short)
        # b is normalization strength
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        term_freq = self.term_frequencies[doc_id][term_token[0]]
        bm25_tf_score = (term_freq * (k1 + 1)) / (term_freq + k1 * length_norm)
        return bm25_tf_score

    # from now on will try to add singnatures
    def get_bm25_idf(self, term: str) -> float:
        term_token = preprocess_text_to_tokens_pipe(term)
        if len(term_token) != 1:
            raise ValueError("InvertedIndex.get_bm25_df() accepts only one term\n")

        doc_freq = len(self.index.get(term_token[0], []))
        # compares documents without term and documents with term
        term_bm25_idf_score = log((len(self.docmap) - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return term_bm25_idf_score

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int=DEFAULT_TOP_K) -> list[dict]:
        query_terms = query.split()
        doc_to_score = {}
        for doc_id in self.docmap:
            bm25_score_sum = 0.0
            for term in query_terms:
                bm25_score_sum += self.bm25(doc_id, term)
            doc_to_score[doc_id] = bm25_score_sum
        
        doc_to_score = list(sorted(doc_to_score.items(), key=lambda it: it[1], reverse=True)[:limit])

        search_results = []
        for doc_id, score in doc_to_score:
            search_results.append({
                "id": doc_id, 
                "title": self.docmap[doc_id]["title"],
                "score": score,
                "description": self.docmap[doc_id]["description"],
                })

        return search_results

    def build(self, docs_path: str=DOCS_JSON_PATH):
        if not os.path.exists(docs_path):
            raise ValueError(f"Json path {docs_path} doesn't exist")
        with open(docs_path, 'r') as f:
            docs_data = json.load(f)["movies"]

        for doc in docs_data:
            doc_text = f"{doc['title']} {doc['description']}"
            self.__add_document(doc["id"], doc_text)
            self.docmap[doc["id"]] = doc

    # saves .index, .docmap, .term_frequencies and .doc_lengths to disk
    def save(self, idx_save_path=INDEX_DB_PATH, docmap_save_path=DOCMAP_PATH, tf_save_path=TERM_FREQ_PATH, doclen_save_path=DOC_LENGTHS_PATH):
        idx_dir_path = os.path.abspath(os.path.dirname(idx_save_path))
        if not os.path.exists(idx_dir_path):
            os.mkdir(idx_dir_path)
        if os.path.exists(idx_save_path):
            raise ValueError(f"{idx_save_path} already exists, delete manually before saving db")
            #os.remove(idx_save_path)

        docmap_dir_path = os.path.abspath(os.path.dirname(docmap_save_path))
        if not os.path.exists(docmap_dir_path):
            os.mkdir(docmap_dir_path)
        if os.path.exists(docmap_save_path):
            raise ValueError(f"{docmap_save_path} already exists, delete manually before saving db")
            #os.remove(docmap_save_path)

        tf_dir_path = os.path.abspath(os.path.dirname(tf_save_path))
        if not os.path.exists(tf_dir_path):
            os.mkdir(tf_dir_path)
        if os.path.exists(tf_save_path):
            raise ValueError(f"{tf_save_path} already exists, delete manually before saving db")
            #os.remove(tf_save_path)
        
        doclen_dir_path = os.path.abspath(os.path.dirname(doclen_save_path))
        if not os.path.exists(doclen_dir_path):
            os.mkdir(doclen_dir_path)
        if os.path.exists(doclen_save_path):
            raise ValueError(f"{doclen_save_path} already exists, delete manually before saving db")
            #os.remove(doclen_save_path)

        pickle.dump(self.index, open(idx_save_path, "wb"))
        pickle.dump(self.docmap, open(docmap_save_path, "wb"))
        pickle.dump(self.term_frequencies, open(tf_save_path, "wb"))
        pickle.dump(self.doc_lengths, open(doclen_save_path, "wb"))

    def load(self, idx_load_path=INDEX_DB_PATH, docmap_load_path=DOCMAP_PATH, tf_load_path=TERM_FREQ_PATH, doclen_load_path=DOC_LENGTHS_PATH):
        if not os.path.exists(idx_load_path):
            raise ValueError(f"Unexisting path to index DB: {idx_load_path}\n")
        if not os.path.exists(docmap_load_path):
            raise ValueError(f"Unexisting path to docmap: {docmap_load_path}\n")
        if not os.path.exists(tf_load_path):
            raise ValueError(f"Unexisting path to term frequencies: {tf_load_path}\n")
        if not os.path.exists(doclen_load_path):
            raise ValueError(f"Unexisting path to documents lengths: {doclen_load_path}\n")

        self.index = pickle.load(open(idx_load_path, "rb"))
        self.docmap = pickle.load(open(docmap_load_path, "rb"))
        self.term_frequencies = pickle.load(open(tf_load_path, "rb"))
        self.doc_lengths = pickle.load(open(doclen_load_path, "rb"))
