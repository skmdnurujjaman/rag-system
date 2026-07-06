import contextvars
import hashlib
import json
import logging
import time

import openai
from langfuse.openai import AsyncOpenAI

from rag.config import settings
from rag.observability import (
    CACHE_HITS,
    MODEL_CALLS,
    MODEL_ERRORS,
    MODEL_LATENCY,
    MODEL_TOKENS,
    log,
    metrics,
    tracer,
)


class CostLimitExceeded(Exception):
    pass

class _Budget:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used = 0
        
_budget: contextvars.ContextVar = contextvars.ContextVar("budget", default=None)


logger = logging.getLogger(__name__)
_llm_api_key= settings.openai_api_key.get_secret_value()
_fallback_llm_api_key= settings.gemini_api_key.get_secret_value()
_client = AsyncOpenAI(api_key=_llm_api_key)

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
CHEAP_MODEL = "gpt-4o-mini"
STRONG_MODEL = "gpt-4o"   # set to "gpt-4o-mini" too if you want zero extra cost while keeping routing

# --- fallback provider: Gemini via its OpenAI-compatible endpoint ---
FALLBACK_MODEL = "gemini-2.5-flash"  # change to any Gemini model your key supports
_fallback_client = AsyncOpenAI(
    api_key=_fallback_llm_api_key,
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

_client = AsyncOpenAI(
    api_key=_llm_api_key,
    timeout=30.0,      # seconds — fail fast instead of hanging
    max_retries=3,     # SDK retries 429/5xx/connection with exponential backoff
)

_COMPLEXITY_SIGNALS = ("compare", "difference", "trade-off", "tradeoff",
                       "versus", " vs ", "step by step", "in detail", "analyze")

def start_budget(max_tokens: int) -> None:
    """Start a per-request token budget (0 = unlimited). Call BEFORE running the graph."""
    _budget.set(_Budget(max_tokens))

def _record_usage(total_tokens: int) -> None:
    b = _budget.get()
    if b is None or b.max_tokens <= 0:
        return
    b.used += total_tokens                      # mutate the shared object
    if b.used > b.max_tokens:
        raise CostLimitExceeded(f"token budget {b.max_tokens} exceeded (used {b.used})")
    
def route_model(query: str) -> str:
    """Pick a model by query complexity: cheap by default, strong for hard queries."""
    is_complex = (
        len(query.split()) > 30
        or query.count("?") > 1
        or any(s in query.lower() for s in _COMPLEXITY_SIGNALS)
    )
    chosen = STRONG_MODEL if is_complex else CHEAP_MODEL
    logger.info("gateway.route words=%d complex=%s -> %s", len(query.split()), is_complex, chosen)
    return chosen

def _make_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()

async def _complete(client: AsyncOpenAI, model: str, messages: list[dict],
              temperature: float, max_tokens: int, kwargs: dict) -> str:
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model", model)
        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )
        tokens = response.usage.total_tokens if response.usage else 0
        _record_usage(tokens)
        latency_ms=round((time.perf_counter() - start) * 1000, 1)
        span.set_attribute("llm.tokens", tokens)
        log.info("llm.call", model=model, tokens=tokens, latency_ms=latency_ms)
        metrics.record_llm_call(latency_ms, tokens)
        MODEL_CALLS.labels(model).inc()
        MODEL_TOKENS.labels(model).inc(tokens)
        MODEL_LATENCY.labels(model).observe(latency_ms / 1000)   # Histograms want seconds

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

async def chat(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0,
         max_tokens: int = 1024, use_cache: bool = True, **kwargs) -> str:
    global _consecutive_failures, _circuit_open_until

    key = _make_key({
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "kwargs": kwargs,
    })
    if use_cache and key in _chat_cache:
        log.info("llm.cache_hit", model=model)
        metrics.record_cache_hit()
        CACHE_HITS.inc()
        return _chat_cache[key]

    logger.info("gateway.chat CACHE MISS model=%s messages=%d", model, len(messages))

    if time.time() >= _circuit_open_until:  # circuit closed -> try primary
        try:
            result = await _complete(_client, model, messages, temperature, max_tokens, kwargs)
            _consecutive_failures = 0                      # success resets
            if use_cache:
                _chat_cache[key] = result
            return result
        except _TRANSIENT as e:
            _consecutive_failures += 1
            metrics.record_error()
            MODEL_ERRORS.inc()
            logger.error("gateway.chat PRIMARY transient failure (%d/%d) error=%s",
                         _consecutive_failures, FAILURE_THRESHOLD, type(e).__name__)
            if _consecutive_failures >= FAILURE_THRESHOLD:
                _circuit_open_until = time.time() + COOLDOWN_SECONDS
                logger.warning("gateway.chat CIRCUIT OPEN %ds -> routing to fallback", COOLDOWN_SECONDS)
        except openai.APIError as e:
            logger.error("gateway.chat PRIMARY client error=%s (no fallback)", type(e).__name__)
            metrics.record_error()
            MODEL_ERRORS.inc()
            raise
    else:
        logger.warning("gateway.chat CIRCUIT OPEN -> skipping primary")

    # --- provider fallback ---
    logger.warning("gateway.chat FALLBACK -> %s", FALLBACK_MODEL)
    result = await _complete(_fallback_client, FALLBACK_MODEL, messages, temperature, max_tokens, kwargs)
    if use_cache:
        _chat_cache[key] = result
    return result


async def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Single entry point for all embedding calls."""
    with tracer.start_as_current_span("llm.embed") as span:
        span.set_attribute("llm.model", model)
        logger.info("gateway.embed model=%s texts=%d", model, len(texts))
        try:
            start = time.perf_counter()
            response = await _client.embeddings.create(model=model, input=texts)
            tokens = response.usage.total_tokens if response.usage else 0
            _record_usage(tokens)
            latency_ms=round((time.perf_counter() - start) * 1000, 1)
            span.set_attribute("llm.model", model)
            log.info("llm.embed", model=model, n_texts=len(texts), tokens=tokens,
                latency_ms=latency_ms)
            metrics.record_embed(latency_ms, tokens)
            MODEL_CALLS.labels(model).inc()
            MODEL_TOKENS.labels(model).inc(tokens)
            MODEL_LATENCY.labels(model).observe(latency_ms / 1000)   # Histograms want seconds
            return [item.embedding for item in response.data]
        except openai.APIError as e:
            metrics.record_error()
            MODEL_ERRORS.inc()
            logger.error("gateway.embed FAILED model=%s error=%s", model, type(e).__name__)
            raise

async def moderate(text: str) -> tuple[bool, list[str]]:
    """Return (flagged, [categories]) via OpenAI's free moderation endpoint."""
    try:
        resp = await _client.moderations.create(model="omni-moderation-latest", input=text)
        r = resp.results[0]
        cats = [k for k, v in r.categories.model_dump().items() if v]
        return r.flagged, cats
    except Exception as e:
        log.warning("moderation.error", error=type(e).__name__)
        return False, []      # fail-open: don't break the app if moderation is down
