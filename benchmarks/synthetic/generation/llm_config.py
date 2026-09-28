"""Credentials for the LLM used to rewrite queries, read from environment variables only.

    LLM_BASE_URL  OpenAI-compatible endpoint (default: a local vLLM server)
    LLM_API_KEY   API key (default: EMPTY)
    LLM_MODEL     model id (the released queries used gemini-3.5-flash)
"""
import os


def llm_client_kwargs():
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("LLM_API_KEY", "EMPTY")
    model = os.environ.get("LLM_MODEL", "gemini-3.5-flash")
    return base_url, api_key, model
