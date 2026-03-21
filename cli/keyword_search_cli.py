#!/usr/bin/env python3

import json
import string
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from nltk.stem import PorterStemmer
from consts import MOVIES_JSON_PATH, STOP_WORDS_PATH


parser = argparse.ArgumentParser(description="Keyword Search CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")

search_parser = subparsers.add_parser("search", help="Search movies using BM25")
search_parser.add_argument("query", type=str, help="Search query")


def remove_punctuation(text) -> str:
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    translation_map = {k: '' for k in string.punctuation}
    translation = str.maketrans(translation_map)
    clear_text = text.translate(translation)
    return clear_text


def match_tokens_count(query_tokens, text_tokens) -> int:
    count = 0
    for qt in query_tokens:
        for tt in text_tokens:
            if qt in tt:
                count += 1
    return count


def clear_tokens_stopwords(text_tokens, stop_words_path=STOP_WORDS_PATH) -> list:
    with open(stop_words_path, 'r') as swf:
        stop_words = swf.read().split()

    clear_tokens = []
    for token in text_tokens:
        if not token in stop_words:
            clear_tokens.append(token)
    
    return clear_tokens


def get_stem_tokens(text_tokens, stemmer) -> list:
    stem_tokens = []
    for token in text_tokens:
        stem_tokens.append(stemmer.stem(token))
    return stem_tokens


def preprocess_text_to_tokens_pipe(text, stemmer) -> list:
    text = text.lower()
    text = remove_punctuation(text)
    text_tokens = text.split()
    text_tokens = clear_tokens_stopwords(text_tokens)
    text_tokens = get_stem_tokens(text_tokens, stemmer)

    return text_tokens



def main() -> None:

    args = parser.parse_args()
    stemmer = PorterStemmer()

    match args.command:
        case "search":
            query_tokens = preprocess_text_to_tokens_pipe(args.query, stemmer)
            found_movies = []
            with open(MOVIES_JSON_PATH, 'r') as f:
                movies_data = json.load(f)["movies"]
            for movie in movies_data:
                movie_tokens = preprocess_text_to_tokens_pipe(movie["title"], stemmer)
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
