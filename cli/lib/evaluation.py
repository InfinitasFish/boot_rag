import json

from consts import DOCS_JSON_PATH, GOLD_DATASET_JSON_PATH, DEFAULT_TOP_K, DEFAULT_RRF_K
from lib.hybrid_search import hybrid_search_init, enhance_query, hybrid_rrf_search, rerank_search_results, hybrid_rrf_res_log


# applying enhance_query and reranking to be able to evaluate each method (ablation study)
def evaluate_rrf_search(limit: int=DEFAULT_TOP_K, k: float=DEFAULT_RRF_K, enhance_method: str=None, rerank_method: str=None, eval_ds_path: str=GOLD_DATASET_JSON_PATH):
    hs = hybrid_search_init()
    
    with open(eval_ds_path, 'r') as f:
        test_cases_data = json.load(f)["test_cases"]

    print(f"RRF Search k = {k}")
    for case in test_cases_data:
        gt = case["relevant_docs"]

        query = enhance_query(case["query"], enhance_method)
        pred = hybrid_rrf_search(hs, query, k, limit)[:limit]
        pred = rerank_search_results(query, pred, rerank_method)

        found_relevant_docs = [doc["title"] for doc in pred if doc["title"] in gt]
        precision_k = len(found_relevant_docs) / len(pred)
        recall_k = len(found_relevant_docs) / len(gt)
        f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision_k:.4f}")
        print(f"  - Recall@{limit}: {recall_k:.4f}")
        print(f"  - F1 Score: {f1_k:.4f}")
        print(f"  - Retrieved: {', '.join([doc['title'] for doc in pred])}")
        print(f"  - Relevant: {', '.join([doc_title for doc_title in found_relevant_docs])}\n")

