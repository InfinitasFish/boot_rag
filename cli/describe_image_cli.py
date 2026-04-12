import argparse
from mimetypes import guess_type
import ollama
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from consts import LLM_VISUAL_MODEL, LLM_TEMPERATURE, LLM_SEED
from prompts import LLM_SYSTEM_PROMPT, MERGE_QUERY_IMAGE


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Processing CLI")
    parser.add_argument("--query", type=str, help="Query to find relevant documents for")
    parser.add_argument("--image", type=str, help="Path to image")

    args = parser.parse_args()
    if not args.query or not args.image:
        parser.print_help()
        return
    
    mime, _ = guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, "rb") as f:
        image_bytes = f.read()
    
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}, {"role": "user", "content": MERGE_QUERY_IMAGE, "images": [image_bytes],}]
    response = ollama.chat(model=LLM_VISUAL_MODEL, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    answer = response["message"]["content"]
    prompt_tokens = response["prompt_eval_count"]
    response_tokens = response["eval_count"]

    print(f"Rewritten query: {answer}")
    print(f"Total tokens: {prompt_tokens + response_tokens}")


if __name__ == "__main__":
    main()
