import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import ollama

from consts import LLM_MODEL, LLM_SEED, LLM_TEMPERATURE


def main():
    parser = argparse.ArgumentParser(description="LLM Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    prompt_question = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
    messages = [{"role": "user", "content": prompt_question}]
    response = ollama.chat(model=LLM_MODEL, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE})

    response_msg_str = response["message"]["content"]
    prompt_tokens = response["prompt_eval_count"]
    response_tokens = response["eval_count"]
    messages.append(response["message"])
    
    print(f"Prompt: {prompt_question}\n{LLM_MODEL} answer: {response_msg_str}\nPrompt tokens: {prompt_tokens}\nResponse tokens: {response_tokens}")


if __name__ == "__main__":
    main()
