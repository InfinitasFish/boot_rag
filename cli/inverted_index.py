import json
import pickle
import os

from consts import MOVIES_JSON_PATH, INDEX_DB_PATH, DOCMAP_PATH
from preprocess import preprocess_text_to_tokens_pipe


# forward index maps location to value,
# an inverted index maps value to location
# e.g. token 'matrix' -> [1, 5, 10] movies ids
class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        text_tokens = preprocess_text_to_tokens_pipe(text)
        for token in text_tokens:
            if token in self.index:
                self.index[token].append(doc_id)
            else:
                self.index[token] = [doc_id]
    
    def get_documents(self, term):
        term = term.lower()
        idx = sorted(self.index[term])
        return idx

    def build(self, movies_path=MOVIES_JSON_PATH):
        with open(MOVIES_JSON_PATH, 'r') as f:
            movies_data = json.load(f)["movies"]

        for movie in movies_data:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_text)
            self.docmap[movie["id"]] = movie

    # saves .index and .docmap to disk
    def save(self, idx_save_path=INDEX_DB_PATH, docmap_save_path=DOCMAP_PATH):
        idx_dir_path = os.path.abspath(os.path.dirname(idx_save_path))
        if not os.path.exists(idx_dir_path):
            os.mkdir(idx_dir_path)

        docmap_dir_path = os.path.abspath(os.path.dirname(docmap_save_path))
        if not os.path.exists(docmap_dir_path):
            os.mkdir(docmap_dir_path)

        pickle.dump(self.index, open(idx_save_path, 'wb'))
        pickle.dump(self.docmap, open(docmap_save_path, 'wb'))
