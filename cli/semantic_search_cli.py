#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from consts import TEXT_EMBEDDING_MODEL
from lib.semantic_search import SemanticSearch, verify_model

parser = argparse.ArgumentParser(description="Semantic Search CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")

model_verify_parser = subparsers.add_parser("verify", help="Download and verify a text embedding model")
model_verify_parser.add_argument("model", type=str, nargs='?', default=TEXT_EMBEDDING_MODEL, help="Model to download from sentence-transformers")


def main():
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model(args.model)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
