import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import json

from consts import DEFAULT_TOP_K
from lib.evaluation import evaluate_rrf_search


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=DEFAULT_TOP_K, help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()

    evaluate_rrf_search(args.limit)


if __name__ == "__main__":
    main()