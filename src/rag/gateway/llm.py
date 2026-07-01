import hashlib
import json
import logging
import time

import openai
from openai import OpenAI

from rag.config import settings

logger = logging.getLogger(__name__)
_client = OpenAI(api_key=settings.openai_api_key)

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# --- fallback provider: Gemini via its OpenAI-compatible endpoint ---
FALLBACK_MODEL = "gemini-2.5-flash"  # change to any Gemini model your key supports
_fallback_client = OpenAI(
    api_key=settings.gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=30.0,
    max_retries=2,
)

# errors that mean "provider is struggling" -> worth failing over
_TRANSIENT = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

# --- circuit breaker state (primary) ---
_consecutive_failures = 0
_circuit_open_until = 0.0
FAILURE_THRESHOLD = 3
COOLDOWN_SECONDS = 30

_chat_cache: dict[str, str] = {}

_client = OpenAI(
    api_key=settings.openai_api_key,
    timeout=30.0,      # seconds — fail fast instead of hanging
    max_retries=3,     # SDK retries 429/5xx/connection with exponential backoff
)

def _make_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()

def _complete(client: OpenAI, model: str, messages: list[dict],
              temperature: float, max_tokens: int, kwargs: dict) -> str:
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, **kwargs,
    )
    return response.choices[0].message.content

def chat_stream(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0,
                max_tokens: int = 1024, **kwargs):
    """Stream the assistant's response token-by-token (a generator)."""
    logger.info("gateway.chat_stream model=%s messages=%d", model, len(messages))
    stream = _client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=True, **kwargs,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

def chat(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0,
         max_tokens: int = 1024, use_cache: bool = True, **kwargs) -> str:
    global _consecutive_failures, _circuit_open_until

    key = _make_key({
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "kwargs": kwargs,
    })
    if use_cache and key in _chat_cache:
        logger.info("gateway.chat CACHE HIT model=%s", model)
        return _chat_cache[key]

    logger.info("gateway.chat CACHE MISS model=%s messages=%d", model, len(messages))

    if time.time() >= _circuit_open_until:  # circuit closed -> try primary
        try:
            result = _complete(_client, model, messages, temperature, max_tokens, kwargs)
            _consecutive_failures = 0                      # success resets
            if use_cache:
                _chat_cache[key] = result
            return result
        except _TRANSIENT as e:
            _consecutive_failures += 1
            logger.error("gateway.chat PRIMARY transient failure (%d/%d) error=%s",
                         _consecutive_failures, FAILURE_THRESHOLD, type(e).__name__)
            if _consecutive_failures >= FAILURE_THRESHOLD:
                _circuit_open_until = time.time() + COOLDOWN_SECONDS
                logger.warning("gateway.chat CIRCUIT OPEN %ds -> routing to fallback", COOLDOWN_SECONDS)
        except openai.APIError as e:                       # client error (4xx) -> our bug
            logger.error("gateway.chat PRIMARY client error=%s (no fallback)", type(e).__name__)
            raise
    else:
        logger.warning("gateway.chat CIRCUIT OPEN -> skipping primary")

    # --- provider fallback ---
    logger.warning("gateway.chat FALLBACK -> %s", FALLBACK_MODEL)
    result = _complete(_fallback_client, FALLBACK_MODEL, messages, temperature, max_tokens, kwargs)
    if use_cache:
        _chat_cache[key] = result
    return result


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Single entry point for all embedding calls."""
    logger.info("gateway.embed model=%s texts=%d", model, len(texts))
    try:
        response = _client.embeddings.create(model=model, input=texts)
    except openai.APIError as e:
        logger.error("gateway.embed FAILED model=%s error=%s", model, type(e).__name__)
        raise

    return [item.embedding for item in response.data]
