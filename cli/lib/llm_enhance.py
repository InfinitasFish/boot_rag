import argparse
import os
import ollama

from consts import LLM_MODEL, LLM_SEED, LLM_TEMPERATURE
from prompts import QUERY_ENHANCE_SYSTEM_PROMPT, QUERY_ENHANCE_SPELL_PROMPTf, QUERY_REWRITE_PROMPTf, QUERY_EXPAND_PROMPTf


def enhance_spelling_user_query(query: str, model: str=LLM_MODEL) -> str:
    enhance_prompt = QUERY_ENHANCE_SPELL_PROMPTf(query)
    messages = [{"role": "system", "content": QUERY_ENHANCE_SYSTEM_PROMPT}, {"role": "user", "content": enhance_prompt}]
    response = ollama.chat(model=model, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    new_query = response["message"]["content"]

    return new_query


def rewrite_user_query(query: str, model: str=LLM_MODEL) -> str:
    rewrite_prompt = QUERY_REWRITE_PROMPTf(query)
    messages = [{"role": "system", "content": QUERY_ENHANCE_SYSTEM_PROMPT}, {"role": "user", "content": rewrite_prompt}]
    response = ollama.chat(model=model, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    new_query = response["message"]["content"]

    return new_query


def expand_user_query(query: str, model: str=LLM_MODEL) -> str:
    expand_prompt = QUERY_EXPAND_PROMPTf(query)
    messages = [{"role": "system", "content": QUERY_ENHANCE_SYSTEM_PROMPT}, {"role": "user", "content": expand_prompt}]
    response = ollama.chat(model=model, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    new_query = response["message"]["content"]

    return new_query
