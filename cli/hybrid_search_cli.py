import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from lib.hybrid_search import minmax_normalize_scores, hybrid_norm_score_search, hybrid_rrf_score_search
from lib.llm_enhance import enhance_spelling_user_query, rewrite_user_query, expand_user_query
from consts import DEFAULT_TOP_K, DEFAULT_ALPHA_WEIGHT, DEFAULT_RRF_K


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores using min-max algorithm")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="List of scores")

    hybrid_norm_search_parser = subparsers.add_parser("weighted-search", help="Search relevant docs using hybrid normalized score (bm25 & semantic)")
    hybrid_norm_search_parser.add_argument("query", type=str, help="Query to find relevant documents for")
    hybrid_norm_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_TOP_K, help="Limit how much documents will contain in result")
    hybrid_norm_search_parser.add_argument("--alpha", type=float, nargs='?', default=DEFAULT_ALPHA_WEIGHT, help="Alpha parameter to weight bm25 & semantic scores")

    hybrid_rrf_search_parser = subparsers.add_parser("rrf-search", help="Search relevant docs using hybrid rrf score (bm25 & semantic)")
    hybrid_rrf_search_parser.add_argument("query", type=str, help="Query to find relevant documents for")
    hybrid_rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    hybrid_rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch"], help="Rerank search results using LLM")
    hybrid_rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_TOP_K, help="Limit how much documents will contain in result")
    hybrid_rrf_search_parser.add_argument("-k", type=float, nargs='?', default=DEFAULT_RRF_K, help="K parameter for calculating RRF score")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_scores = minmax_normalize_scores(args.scores)
            for score in norm_scores:
                print(f" * {score:.4f}")
        case "weighted-search":
            hybrid_norm_score_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            query = args.query
            if args.enhance == "spell":
                query = enhance_spelling_user_query(args.query)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
            elif args.enhance == "rewrite":
                query = rewrite_user_query(args.query)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
            elif args.enhance == "expand":
                query = expand_user_query(args.query)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
            hybrid_rrf_score_search(query, args.k, args.limit, args.rerank_method)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
