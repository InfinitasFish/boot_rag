#!/usr/bin/env python3

import json
import string
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from consts import MOVIES_JSON_PATH, STOP_WORDS_PATH, INDEX_DB_PATH, DOCMAP_PATH
from preprocess import preprocess_text_to_tokens_pipe, match_tokens_count
from inverted_index import InvertedIndex

parser = argparse.ArgumentParser(description="Keyword Search CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")

search_parser = subparsers.add_parser("search", help="Search movies using BM25")
search_parser.add_argument("query", type=str, help="Search query")

build_parser = subparsers.add_parser("build", help="Build Index DB for movies")


# adapting new "" quotes for strings instead of '' from now on
def main() -> None:

    args = parser.parse_args()
    idx_db = InvertedIndex()

    match args.command:
        case "build":
            idx_db.build()
            idx_db.save()
            print(f"Index DB build done.")
        case "search":
            idx_db.load()
            query_tokens = preprocess_text_to_tokens_pipe(args.query)
            found_movies_ids = []
            for token in query_tokens:
                if token in idx_db.index:
                    found_movies_ids.extend(idx_db.index[token])
            for i, ids in enumerate(found_movies_ids):
                if i >= 5: break
                print(f"{i+1}. {idx_db.docmap[ids]['title']} {ids}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
