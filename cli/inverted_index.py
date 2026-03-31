import json
import pickle
import os
from collections import Counter
# import sys
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from consts import MOVIES_JSON_PATH, INDEX_DB_PATH, DOCMAP_PATH, TERM_FREQ_PATH
from preprocess import preprocess_text_to_tokens_pipe


# forward index maps location to value,
# an inverted index maps value to location
# e.g. token 'matrix' -> [1, 5, 10] movies ids
class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        text_tokens = preprocess_text_to_tokens_pipe(text)
        self.term_frequencies[doc_id] = Counter(text_tokens)
        for token in self.term_frequencies[doc_id]:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = set([doc_id])
    
    def get_documents(self, term):
        term = term.lower()
        idx = sorted(self.index[term])
        return idx
    
    def get_tf(self, doc_id, term):
        token = preprocess_text_to_tokens_pipe(term)
        if len(token) != 1:
            raise ValueError("InvertedIndex.get_tf() accepts only one term\n")
        return self.term_frequencies[doc_id][token[0]]

    def build(self, movies_path=MOVIES_JSON_PATH):
        with open(MOVIES_JSON_PATH, 'r') as f:
            movies_data = json.load(f)["movies"]

        for movie in movies_data:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_text)
            self.docmap[movie["id"]] = movie

    # saves .index, .docmap and .term_frequencies to disk
    def save(self, idx_save_path=INDEX_DB_PATH, docmap_save_path=DOCMAP_PATH, tf_save_path=TERM_FREQ_PATH):
        idx_dir_path = os.path.abspath(os.path.dirname(idx_save_path))
        if not os.path.exists(idx_dir_path):
            os.mkdir(idx_dir_path)
        if os.path.exists(idx_save_path):
            os.remove(idx_save_path)

        docmap_dir_path = os.path.abspath(os.path.dirname(docmap_save_path))
        if not os.path.exists(docmap_dir_path):
            os.mkdir(docmap_dir_path)
        if os.path.exists(docmap_save_path):
            os.remove(docmap_save_path)

        tf_dir_path = os.path.abspath(os.path.dirname(tf_save_path))
        if not os.path.exists(tf_dir_path):
            os.mkdir(tf_dir_path)
        if os.path.exists(tf_save_path):
            os.remove(tf_save_path)

        pickle.dump(self.index, open(idx_save_path, "wb"))
        pickle.dump(self.docmap, open(docmap_save_path, "wb"))
        pickle.dump(self.term_frequencies, open(tf_save_path, "wb"))

    def load(self, idx_load_path=INDEX_DB_PATH, docmap_load_path=DOCMAP_PATH, tf_load_path=TERM_FREQ_PATH):
        if not os.path.exists(idx_load_path):
            raise ValueError(f"Unexisting path to index DB: {idx_load_path}\n")
        if not os.path.exists(docmap_load_path):
            raise ValueError(f"Unexisting path to docmap: {docmap_load_path}\n")
        if not os.path.exists(tf_load_path):
            raise ValueError(f"Unexisting path to term frequencies: {tf_load_path}\n")

        self.index = pickle.load(open(idx_load_path, "rb"))
        self.docmap = pickle.load(open(docmap_load_path, "rb"))
        self.term_frequencies = pickle.load(open(tf_load_path, "rb"))

