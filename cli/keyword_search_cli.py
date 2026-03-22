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


def main() -> None:

    args = parser.parse_args()
    idx_db = InvertedIndex()

    match args.command:
        case "build":
            idx_db.build()
            idx_db.save()
            test_token = "merida"
            token_movies = idx_db.index[test_token]
            print(f"Index Db build done. First document for token '{test_token}' = {token_movies[0]}")

        case "search":
            query_tokens = preprocess_text_to_tokens_pipe(args.query)
            found_movies = []
            with open(MOVIES_JSON_PATH, 'r') as f:
                movies_data = json.load(f)["movies"]
            for movie in movies_data:
                movie_tokens = preprocess_text_to_tokens_pipe(movie["title"])
                if match_tokens_count(query_tokens, movie_tokens) > 0:
                    found_movies.append(movie)

            found_movies = sorted(found_movies, key=lambda d: d["id"])
            for i, movie in enumerate(found_movies):
                if i > 4:
                    break
                print(f"{i+1}. {movie['title']} {movie['id']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
