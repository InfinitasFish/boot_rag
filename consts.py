import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_JSON_PATH = os.path.join(ROOT_DIR, "data/movies.json") 
STOP_WORDS_PATH = os.path.join(ROOT_DIR, "data/stopwords.txt")
INDEX_DB_PATH = os.path.join(ROOT_DIR, "cache/index.pkl")
DOCMAP_PATH = os.path.join(ROOT_DIR, "cache/docmap.pkl")
TERM_FREQ_PATH = os.path.join(ROOT_DIR, "cache/term_freq.pkl")
DOC_LENGTHS_PATH = os.path.join(ROOT_DIR, "cache/doc_lengths.pkl")

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BM25_K1 = 1.5
BM25_B = 0.75
