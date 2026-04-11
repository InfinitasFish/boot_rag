import argparse
import os
import ollama
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from consts import DEFAULT_DESCRIPTION_LEN, LLM_MODEL, LLM_SEED, LLM_TEMPERATURE, CROSS_RERANKER_MODEL
from prompts import LLM_ENHANCE_SYSTEM_PROMPT, QUERY_ENHANCE_SPELL_PROMPTf, QUERY_REWRITE_PROMPTf, QUERY_EXPAND_PROMPTf, RERANK_SEARCH_RESULTSf, BATCH_RERANK_SEARCH_RESULTSf


class IndividualScoreOutput(BaseModel):
    score: float

class BatchRerankOutput(BaseModel):
    ranks: list[int]


def enhance_spelling_user_query(query: str, model: str=LLM_MODEL) -> str:
    enhance_prompt = QUERY_ENHANCE_SPELL_PROMPTf(query)
    messages = [{"role": "system", "content": LLM_ENHANCE_SYSTEM_PROMPT}, {"role": "user", "content": enhance_prompt}]
    response = ollama.chat(model=model, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    new_query = response["message"]["content"]

    return new_query


def rewrite_user_query(query: str, model: str=LLM_MODEL) -> str:
    rewrite_prompt = QUERY_REWRITE_PROMPTf(query)
    messages = [{"role": "system", "content": LLM_ENHANCE_SYSTEM_PROMPT}, {"role": "user", "content": rewrite_prompt}]
    response = ollama.chat(model=model, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    new_query = response["message"]["content"]

    return new_query


def expand_user_query(query: str, model: str=LLM_MODEL) -> str:
    expand_prompt = QUERY_EXPAND_PROMPTf(query)
    messages = [{"role": "system", "content": LLM_ENHANCE_SYSTEM_PROMPT}, {"role": "user", "content": expand_prompt}]
    response = ollama.chat(model=model, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    new_query = response["message"]["content"]

    return new_query


def rerank_search_results(query: str, search_results: list[dict], model: str=LLM_MODEL) -> list[tuple]:
    idx_to_rank_score = {}
    for i, doc in enumerate(search_results):
        score_prompt = RERANK_SEARCH_RESULTSf(query, doc)
        messages = [{"role": "system", "content": LLM_ENHANCE_SYSTEM_PROMPT}, {"role": "user", "content": score_prompt}]
        response = ollama.chat(model=model, messages=messages, format=IndividualScoreOutput.model_json_schema(), options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
        try:
            doc_score = IndividualScoreOutput.model_validate_json(response["message"]["content"]).score
        except ValueError as e:
            doc_score = 0.0
            print(e)
            print(f"Assigning '{doc}' rank score to zero")
        idx_to_rank_score[i] = doc_score

    idx_to_rank_score = sorted(list(idx_to_rank_score.items()), key=lambda it: it[1], reverse=True)
    return idx_to_rank_score


def batch_rerank_search_results(query: str, search_results: list[dict], model: str=LLM_MODEL) -> list[int]:
    docs_str = '\n'.join([f"{i}. {doc.get('title', '')} : {doc.get('description', '')[:DEFAULT_DESCRIPTION_LEN]}" for i, doc in enumerate(search_results)])
    batch_score_prompt = BATCH_RERANK_SEARCH_RESULTSf(query, docs_str)
    messages = [{"role": "system", "content": LLM_ENHANCE_SYSTEM_PROMPT}, {"role": "user", "content": batch_score_prompt}]
    response = ollama.chat(model=model, messages=messages, format=BatchRerankOutput.model_json_schema(), options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    ranked_idxs = []
    try:
        # "[1, 2, 3]" -> [1, 2, 3]
        # ranked_idxs = response["message"]["content"].strip()[1:-1].split(',')
        ranked_idxs = BatchRerankOutput.model_validate_json(response["message"]["content"]).ranks
        print(len(search_results), len(ranked_idxs), '\n', ranked_idxs)
        assert len(ranked_idxs) == len(search_results)
    except ValueError as e:
        print(e)
        print(f"Docs ranks weren't changed")
        ranked_idxs = [i for i in range(len(search_results))]

    return ranked_idxs


def cross_encoder_rerank_search_results(query: str, search_results: list[dict], model: str=CROSS_RERANKER_MODEL) -> list[int]:
    # https://www.sbert.net/docs/package_reference/cross_encoder/model.html
    query_doc_pairs = [(query, f"{doc.get('title', '')} - {doc.get('description', '')}") for doc in search_results]
    ce = CrossEncoder(model_name_or_path=model)
    # scores are aligned with .predict() input
    scores = ce.predict(query_doc_pairs)
    sorted_relevant_search_results = sorted(list(zip(list(range(len(scores))), scores)), key=lambda it: it[1], reverse=True)

    ranked_idxs = [idx for idx, _ in sorted_relevant_search_results]
    return ranked_idxs
