import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from lib.hybrid_search import hybrid_search_init, hybrid_rrf_search
from lib.augmented_generation import rag_answer_question, rag_answer_log, rag_summarize_results, rag_summarization_log, rag_answer_wcitations
from consts import DEFAULT_TOP_K, DEFAULT_RRF_K


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_question_parser = subparsers.add_parser("question", help="Perform RAG (search + generate answer)")
    rag_question_parser.add_argument("query", type=str, help="Query to find relevant documents for")
    rag_question_parser.add_argument("question", type=str, nargs='?', help="Question query for RAG")
    rag_question_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_TOP_K, help="Limit how much documents will contain in result")
    rag_question_parser.add_argument("-k", type=float, nargs='?', default=DEFAULT_RRF_K, help="K parameter for calculating RRF score")

    rag_summarize_parser = subparsers.add_parser("summarize", help="Perform RAG multi-document summarization")
    rag_summarize_parser.add_argument("query", type=str, help="Query to find relevant documents for")
    rag_summarize_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_TOP_K, help="Limit how much documents will contain in result")
    rag_summarize_parser.add_argument("-k", type=float, nargs='?', default=DEFAULT_RRF_K, help="K parameter for calculating RRF score")

    rag_citations_parser = subparsers.add_parser("citations", help="Perform RAG multi-document summarization")
    rag_citations_parser.add_argument("query", type=str, help="Query to find relevant documents for")
    rag_citations_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_TOP_K, help="Limit how much documents will contain in result")
    rag_citations_parser.add_argument("-k", type=float, nargs='?', default=DEFAULT_RRF_K, help="K parameter for calculating RRF score")

    args = parser.parse_args()

    match args.command:
        case "question":
            hs = hybrid_search_init()
            query = args.query
            question = args.question
            search_res = hybrid_rrf_search(hs, query, args.k, args.limit)[:args.limit]
            if question is None:
                question = query
            answer = rag_answer_question(question, search_res)
            rag_answer_log(answer, search_res)
        case "summarize":
            hs = hybrid_search_init()
            query = args.query
            search_res = hybrid_rrf_search(hs, query, args.k, args.limit)[:args.limit]
            summarization = rag_summarize_results(query, search_res)
            rag_summarization_log(summarization, search_res)
        case "citations":
            hs = hybrid_search_init()
            query = args.query
            search_res = hybrid_rrf_search(hs, query, args.k, args.limit)[:args.limit]
            answer = rag_answer_wcitations(args.query, search_res)
            rag_wcitations_log(answer, search_res)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

