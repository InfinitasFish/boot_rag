import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_JSON_PATH = os.path.join(ROOT_DIR, "data/movies.json") 
STOP_WORDS_PATH = os.path.join(ROOT_DIR, "data/stopwords.txt")
INDEX_DB_PATH = os.path.join(ROOT_DIR, "cache/index.pkl")
DOCMAP_PATH = os.path.join(ROOT_DIR, "cache/docmap.pkl")
TERM_FREQ_PATH = os.path.join(ROOT_DIR, "cache/term_freq.pkl")