import argparse
import os
import ollama

from consts import LLM_MODEL, LLM_SEED, LLM_TEMPERATURE
from prompts import QUERY_ENHANCE_PROMPT


def enhance_user_query(query: str, model: str=LLM_MODEL) -> str:
    enhance_prompt = QUERY_ENHANCE_PROMPT + f"\nUser query: \"{query}\""
    messages = [{"role": "user", "content": enhance_prompt}]
    response = ollama.chat(model=model, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE})
    new_query = response["message"]["content"]

    return new_query
