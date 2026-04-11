import json

from consts import DOCS_JSON_PATH, GOLD_DATASET_JSON_PATH, DEFAULT_TOP_K, DEFAULT_RRF_K
from lib.hybrid_search import hybrid_rrf_score_search


def evaluate_rrf_search(limit: int=DEFAULT_TOP_K, k: float=DEFAULT_RRF_K, docs_json_path: str=DOCS_JSON_PATH, eval_ds_path: str=GOLD_DATASET_JSON_PATH):
    with open(eval_ds_path, 'r') as f:
        test_cases_data = json.load(f)["test_cases"]

    print(f"RRF Search k = {k}")
    for case in test_cases_data:
        query = case["query"]
        gt = case["relevant_docs"]
        pred = hybrid_rrf_score_search(query, k, limit)
        found_relevant_docs = [doc["title"] for doc in pred if doc["title"] in gt]
        precision_k = len(found_relevant_docs) / len(pred)
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision_k:.4f}")
        print(f"  - Retrieved: {', '.join([doc['title'] for doc in pred])}")
        print(f"  - Relevant: {', '.join([doc_title for doc_title in found_relevant_docs])}")

