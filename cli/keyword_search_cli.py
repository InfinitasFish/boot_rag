#!/usr/bin/env python3

import json
import string
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from consts import DOC_LENGTHS_PATH, STOP_WORDS_PATH, INDEX_DB_PATH, DOCMAP_PATH, DEFAULT_DESCRIPTION_LEN, BM25_K1, BM25_B
from lib.preprocess import preprocess_text_to_tokens_pipe, match_tokens_count
from lib.inverted_index import InvertedIndex


parser = argparse.ArgumentParser(description="Keyword Search CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")

search_parser = subparsers.add_parser("search", help="Search movies using TF-IDF / BM25")
search_parser.add_argument("query", type=str, help="Search query")

build_parser = subparsers.add_parser("build", help="Build Index DB for movies")

term_freq_parser = subparsers.add_parser("tf", help="Get frequency of a term in movie")
term_freq_parser.add_argument("doc_id", type=int, help="Document ID")
term_freq_parser.add_argument("term", type=str, help="Term to get frequency of in movie")

idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency score for a given term")
idf_parser.add_argument("term", type=str, help="Term to get IDF score of")

# tf-idf is a bit unstable for rare/common terms; doesn't account for the document length
tf_idf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF score for a given doc_id and a term")
tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
tf_idf_parser.add_argument("term", type=str, help="Term to get TF-IDF score of")

bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

bm25_idf_parser = subparsers.add_parser("bm25idf", help="Calculate BM25 IDF score for a given term")
bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
bm25_search_parser.add_argument("query", type=str, help="Search query")

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
            doc_title = idx_db.docmap.get(args.doc_id, {"title": ''})["title"]
            print(f"Default TF score of '{args.term}' in document '{doc_title} ({args.doc_id})': {idx_db.get_tf(args.doc_id, args.term):.2f}")
        case "idf":
            idx_db.load()
            print(f"Inverse document frequency score of '{args.term}': {idx_db.get_idf(args.term):.2f}")
        case "tfidf":
            idx_db.load()
            doc_title = idx_db.docmap.get(args.doc_id, {"title": ''})["title"]
            print(f"TF-IDF score of '{args.term}' in document '{doc_title} ({args.doc_id})': {idx_db.get_tf_idf(args.doc_title, args.term):.2f}")
        case "bm25tf":
            idx_db.load()
            doc_title = idx_db.docmap.get(args.doc_id, {"title": ''})["title"]
            print(f"BM25 TF score of '{args.term}' in document '{doc_title} ({args.doc_id})': {idx_db.get_bm25_tf(args.doc_id, args.term, args.k1, args.b):.2f}")
        case "bm25idf":
            idx_db.load()
            print(f"BM25 IDF score of '{args.term}': {idx_db.get_bm25_idf(args.term):.2f}")
        case "bm25search":
            idx_db.load()
            top_k_docs = idx_db.bm25_search(args.query)
            for i, doc in enumerate(top_k_docs):
                print(f"{i + 1}. ({doc['id']}) {doc['title']} - Score: {doc['score']:.2f}\n  {doc['description'][:DEFAULT_DESCRIPTION_LEN]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
