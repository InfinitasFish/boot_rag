import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from consts import CLIP_MODEL
from lib.multimodal_search import verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_img_embedding = subparsers.add_parser("verify_image_embedding", help="Download and verify an image embedding model")
    verify_img_embedding.add_argument("image", type=str, help="Path to image to verify embeddings with")
    verify_img_embedding.add_argument("--model", type=str, nargs='?', default=CLIP_MODEL, help="Image embedding model")

    args = parser.parse_args()
    
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image, args.model)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
