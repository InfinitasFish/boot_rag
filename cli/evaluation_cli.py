import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import json

from consts import DEFAULT_TOP_K, DEFAULT_RRF_K
from lib.evaluation import evaluate_rrf_search


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("-k", type=float, nargs='?', default=DEFAULT_RRF_K, help="K parameter for calculating RRF score")
    parser.add_argument("--limit", type=int, default=DEFAULT_TOP_K, help="Number of results to evaluate (k for precision@k, recall@k)")
    parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank search results using LLM")

    args = parser.parse_args()

    evaluate_rrf_search(args.limit, args.k, args.enhance, args.rerank_method)


if __name__ == "__main__":
    main()