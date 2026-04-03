#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from consts import DOCS_JSON_PATH, TEXT_EMBEDDING_MODEL, DEFAULT_TOP_K
from lib.semantic_search import SemanticSearch, verify_model, verify_embeddings, embed_text, embed_query_text, search

parser = argparse.ArgumentParser(description="Semantic Search CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")

model_verify_parser = subparsers.add_parser("verify", help="Download and verify a text embedding model")
model_verify_parser.add_argument("model", type=str, nargs='?', default=TEXT_EMBEDDING_MODEL, help="Model to use from sentence-transformers")

gen_emb_parser = subparsers.add_parser("embed_text", help="Generate embedding for text")
gen_emb_parser.add_argument("model", type=str, nargs='?', default=TEXT_EMBEDDING_MODEL, help="Model to use from sentence-transformers")
gen_emb_parser.add_argument("text", type=str, help="Text to generate embedding for")

emb_verify_parser = subparsers.add_parser("verify_embeddings", help="Load and verify documents embeddings")
emb_verify_parser.add_argument("model", type=str, nargs='?', default=TEXT_EMBEDDING_MODEL, help="Model to use from sentence-transformers")
emb_verify_parser.add_argument("json_path", type=str, nargs='?', default=DOCS_JSON_PATH, help="Json path to get docs from")

emb_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for a query")
emb_query_parser.add_argument("model", type=str, nargs='?', default=TEXT_EMBEDDING_MODEL, help="Model to use from sentence-transformers")
emb_query_parser.add_argument("query", type=str, help="Query to generate embedding for")

semantic_search_parser = subparsers.add_parser("search", help="Use semantic search to find most relevant documents")
semantic_search_parser.add_argument("model", type=str, nargs='?', default=TEXT_EMBEDDING_MODEL, help="Model to use from sentence-transformers")
semantic_search_parser.add_argument("query", type=str, help="Query to find relevant documents for")
semantic_search_parser.add_argument("-l", "--limit", type=int, nargs='?', default=DEFAULT_TOP_K, help="Limit how much documents will contain in result")


def main():
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model(args.model)
        case "embed_text":
            embed_text(args.text, args.model)
        case "verify_embeddings":
            verify_embeddings(args.model, args.json_path)
        case "embedquery":
            embed_query_text(args.query, args.model)
        case "search":
            search(args.query, args.limit, args.model)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
