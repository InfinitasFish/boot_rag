#!/usr/bin/env python3

import json
import string
import argparse
import os
from math import log
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

term_freq_parser = subparsers.add_parser("tf", help="Get frequency of term in movie")
term_freq_parser.add_argument("movie_id", type=int, help="Movie id")
term_freq_parser.add_argument("term", type=str, help="Term to get frequency of in movie")

idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency score for a term")
idf_parser.add_argument("term", type=str, help="Term to get idf score of")

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
        case "tf":
            idx_db.load()
            if args.movie_id in idx_db.docmap:
                print(f"Movie: {idx_db.docmap[args.movie_id]['title']}; Term-freq: {args.term} {idx_db.get_tf(args.movie_id, args.term)}")
            else:
                print(f"Unknown movie_id {args.movie_id}")
        case "idf":
            idx_db.load()
            term_token = preprocess_text_to_tokens_pipe(args.term)
            if len(term_token) != 1:
                raise ValueError("Idf accepts only one term\n")

            # avoid zero-division
            appearance_count = len(idx_db.index.get(term_token[0], []))
            term_idf_score = log((len(idx_db.docmap) + 1) / (appearance_count + 1))
            print(f"Inverse document frequency score of '{args.term}': {term_idf_score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
