import hashlib
import json
import logging

from openai import OpenAI

from rag.config import settings

logger = logging.getLogger(__name__)
_client = OpenAI(api_key=settings.openai_api_key)

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

_chat_cache: dict[str, str] = {}


def _make_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def chat(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0,
         max_tokens: int = 1024, use_cache: bool = True, **kwargs) -> str:
    key = _make_key({
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "kwargs": kwargs,
    })
    if use_cache and key in _chat_cache:
        logger.info("gateway.chat CACHE HIT model=%s", model)
        return _chat_cache[key]

    logger.info("gateway.chat CACHE MISS model=%s messages=%d", model, len(messages))
    response = _client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, **kwargs,
    )
    result = response.choices[0].message.content
    if use_cache:
        _chat_cache[key] = result
    return result


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Single entry point for all embedding calls."""
    logger.info("gateway.embed model=%s texts=%d", model, len(texts))
    response = _client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
