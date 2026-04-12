import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from lib.hybrid_search import minmax_normalize_scores, hybrid_search_init, hybrid_norm_search, hybrid_norm_res_log, enhance_query, hybrid_rrf_search, rerank_search_results, hybrid_rrf_res_log
from lib.llm_enhance import enhance_spelling_user_query, rewrite_user_query, expand_user_query, judge_search_results, judge_scores_log
from consts import DEFAULT_TOP_K, DEFAULT_ALPHA_WEIGHT, DEFAULT_RRF_K


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores using min-max algorithm")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="List of scores")

    hybrid_norm_search_parser = subparsers.add_parser("weighted-search", help="Search relevant docs using hybrid normalized score (bm25 & semantic)")
    hybrid_norm_search_parser.add_argument("query", type=str, help="Query to find relevant documents for")
    hybrid_norm_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    hybrid_norm_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank search results using LLM")
    hybrid_norm_search_parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, help="Use and LLM to evaluate the search results")
    hybrid_norm_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_TOP_K, help="Limit how much documents will contain in result")
    hybrid_norm_search_parser.add_argument("--alpha", type=float, nargs='?', default=DEFAULT_ALPHA_WEIGHT, help="Alpha parameter to weight bm25 & semantic scores")

    hybrid_rrf_search_parser = subparsers.add_parser("rrf-search", help="Search relevant docs using hybrid rrf score (bm25 & semantic)")
    hybrid_rrf_search_parser.add_argument("query", type=str, help="Query to find relevant documents for")
    hybrid_rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    hybrid_rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank search results using LLM")
    hybrid_rrf_search_parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, help="Use and LLM to evaluate the search results")
    hybrid_rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_TOP_K, help="Limit how much documents will contain in result")
    hybrid_rrf_search_parser.add_argument("-k", type=float, nargs='?', default=DEFAULT_RRF_K, help="K parameter for calculating RRF score")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_scores = minmax_normalize_scores(args.scores)
            for score in norm_scores:
                print(f" * {score:.4f}")
        case "weighted-search":
            hs = hybrid_search_init()

            query = enhance_query(args.query)
            if args.enhance is not None:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
    
            search_res = hybrid_norm_search(hs, args.query, args.alpha, args.limit)
            # print("Results before reranking:\n")
            # hybrid_norm_res_log(search_res)

            if args.rerank_method is not None:
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
            search_res = rerank_search_results(query, search_res, args.rerank_method)[:args.limit]
    
            hybrid_norm_res_log(search_res)
        case "rrf-search":
            hs = hybrid_search_init()

            query = enhance_query(args.query)
            if args.enhance is not None:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            search_res = hybrid_rrf_search(hs, query, args.k, args.limit)
            # print("Results before reranking:\n")
            # hybrid_rrf_res_log(search_res)

            if args.rerank_method is not None:
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
            search_res = rerank_search_results(query, search_res, args.rerank_method)[:args.limit]

            # call judge after reranking
            if args.evaluate:
                llm_scores = judge_search_results(query, search_res)
                judge_scores_log(llm_scores, search_res)

            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
            hybrid_rrf_res_log(search_res)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
