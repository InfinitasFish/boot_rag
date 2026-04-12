import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_JSON_PATH = os.path.join(ROOT_DIR, "data/movies.json")
GOLD_DATASET_JSON_PATH = os.path.join(ROOT_DIR, "data/golden_dataset.json")
STOP_WORDS_PATH = os.path.join(ROOT_DIR, "data/stopwords.txt")
INDEX_DB_PATH = os.path.join(ROOT_DIR, "cache/index.pkl")
DOCMAP_PATH = os.path.join(ROOT_DIR, "cache/docmap.pkl")
TERM_FREQ_PATH = os.path.join(ROOT_DIR, "cache/term_freq.pkl")
DOC_LENGTHS_PATH = os.path.join(ROOT_DIR, "cache/doc_lengths.pkl")

DEFAULT_DESCRIPTION_LEN = 200
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_OVERLAP_SIZE = 0
DEFAULT_SEARCH_OVERLAP_SIZE = 1
DEFAULT_ALPHA_WEIGHT = 0.2
DEFAULT_RRF_K = 60

LLM_SEED = 59
LLM_TEMPERATURE = 0.1

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
LLM_MODEL = "qwen3:4b"
LLM_VISUAL_MODEL = "qwen3-vl:4b"
CLIP_MODEL = "clip-ViT-B-32"

EMBEDDINGS_SAVE_PATH = os.path.join(ROOT_DIR, "cache/doc_embeddings.npy")
CLIP_TEXT_EMBEDDINGS_SAVE_PATH = os.path.join(ROOT_DIR, "cache/clip_doc_embeddings.npy")
CHUNK_EMBEDDINGS_SAVE_PATH = os.path.join(ROOT_DIR, "cache/doc_chunk_embeddings.npy")
CHUNK_META_SAVE_PATH = os.path.join(ROOT_DIR, "cache/doc_chunk_meta.json")
