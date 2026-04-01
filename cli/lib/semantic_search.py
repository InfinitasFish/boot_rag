from sentence_transformers import SentenceTransformer

from consts import TEXT_EMBEDDING_MODEL


def verify_model(model=TEXT_EMBEDDING_MODEL):
    # dont streSS
    ss = SemanticSearch(model=model)
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


class SemanticSearch:
    def __init__(self, model=TEXT_EMBEDDING_MODEL):
        self.model = SentenceTransformer(model)

