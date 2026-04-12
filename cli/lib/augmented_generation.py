import json
import ollama

from consts import DEFAULT_DESCRIPTION_LEN, LLM_MODEL, LLM_TEMPERATURE, LLM_SEED
from prompts import LLM_SYSTEM_PROMPT, RAG_ANSWER_QUESTIONf, RAG_SUMMARIZE_RESULTSf, RAG_ANSWER_wCITATIONSf


def rag_answer_question(question: str, search_results: list[dict], model: str=LLM_MODEL) -> str:
    docs_str = '\n'.join([f"{i}. {doc.get('title', '')} : {doc.get('description', '')[:DEFAULT_DESCRIPTION_LEN * 2]}" for i, doc in enumerate(search_results)])
    question_prompt = RAG_ANSWER_QUESTIONf(question, docs_str)
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}, {"role": "user", "content": question_prompt}]
    response = ollama.chat(model=model, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    answer = response["message"]["content"]
    return answer


def rag_answer_log(answer: str, search_results: list[dict]):
    print("Search Results:")
    for doc in search_results:
        print(f"- {doc['title']}")
    print(f"\nLLM Answer:\n{answer}\n")


def rag_summarize_results(query: str, search_results: list[dict], model: str=LLM_MODEL) -> str:
    docs_str = '\n'.join([f"{i}. {doc.get('title', '')} : {doc.get('description', '')[:DEFAULT_DESCRIPTION_LEN * 2]}" for i, doc in enumerate(search_results)])
    summarize_prompt = RAG_SUMMARIZE_RESULTSf(query, docs_str)
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}, {"role": "user", "content": summarize_prompt}]
    response = ollama.chat(model=model, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    summarization = response["message"]["content"]
    return summarization


def rag_summarization_log(summarization: str, search_results: list[dict]):
    print("Search Results:")
    for doc in search_results:
        print(f"- {doc['title']}")
    print(f"\nLLM Summary:\n{summarization}\n")


def rag_answer_wcitations(query: str, search_results: list[dict], model: str=LLM_MODEL) -> str:
    docs_str = '\n'.join([f"{i}. {doc.get('title', '')} : {doc.get('description', '')[:DEFAULT_DESCRIPTION_LEN * 2]}" for i, doc in enumerate(search_results)])
    citations_prompt = RAG_ANSWER_wCITATIONSf(query, docs_str)
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}, {"role": "user", "content": citations_prompt}]
    response = ollama.chat(model=model, messages=messages, options={"seed": LLM_SEED, "temperature": LLM_TEMPERATURE,})
    answer = response["message"]["content"]
    return answer

