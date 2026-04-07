import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from lib.hybrid_search import minmax_normalize_scores, Hyb
from consts import DEFAULT_TOP_K, DEFAULT_ALPHA_WEIGHT


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores using min-max algorithm")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="List of scores")

    hybrid_search_parser = subparsers.add_parser("weighted_search", help="Search relevant docs using hybrid score (bm25 & semantic)")
    hybrid_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_TOP_K, help="Limit how much documents will contain in result")
    hybrid_search_parser.add_argument("--alpha", type=float, nargs='?', default=DEFAULT_ALPHA_WEIGHT, help="Alpha to weight bm25 & semantic scores")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_scores = minmax_normalize_scores(args.scores)
            for score in norm_scores:
                print(f" * {score:.4f}")
        case "weighted_search":
            search_results = 
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
