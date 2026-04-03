#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from consts import DOCS_JSON_PATH, TEXT_EMBEDDING_MODEL
from lib.semantic_search import SemanticSearch, verify_model, verify_embeddings, embed_text

parser = argparse.ArgumentParser(description="Semantic Search CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")

model_verify_parser = subparsers.add_parser("verify", help="Download and verify a text embedding model")
model_verify_parser.add_argument("model", type=str, nargs='?', default=TEXT_EMBEDDING_MODEL, help="Model to download from sentence-transformers")

gen_emb_parser = subparsers.add_parser("embed_text", help="Generate embedding for text")
gen_emb_parser.add_argument("model", type=str, nargs='?', default=TEXT_EMBEDDING_MODEL, help="Model to download from sentence-transformers")
gen_emb_parser.add_argument("text", type=str, help="Text to generate embedding for")

emb_verify_parser = subparsers.add_parser("verify_embeddings", help="Load and verify documents embeddings")
emb_verify_parser.add_argument("model", type=str, nargs='?', default=TEXT_EMBEDDING_MODEL, help="Model to download from sentence-transformers")
emb_verify_parser.add_argument("json_path", type=str, nargs='?', default=DOCS_JSON_PATH, help="Json path to get docs from")


def main():
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model(args.model)
        case "embed_text":
            embed_text(args.text, args.model)
        case "verify_embeddings":
            verify_embeddings(args.model, args.json_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
