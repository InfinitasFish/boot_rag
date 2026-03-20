#!/usr/bin/env python3

import json
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Serach CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query.lower()
            found_movies = []
            with open('data/movies.json', 'r') as f:
                movies_data = json.load(f)['movies']
            for movie in movies_data:
                if query in ([movie['title'].lower()] + movie['title'].lower().split()):
                    found_movies.append(movie)

            found_movies = sorted(found_movies, key=lambda d: d['id'])
            for i, movie in enumerate(found_movies):
                if i > 4:
                    break
                print(f"{i+1}. {movie['title']} {movie['id']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
