import os
import structlog
import threading

from rag.config import settings
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from prometheus_client import Counter, Histogram

if settings.langfuse_public_key:
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key.get_secret_value()
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))  # prints spans; swap for OTLP later
trace.set_tracer_provider(_provider)

tracer = trace.get_tracer("rag")

structlog.configure(
    processors=[
        structlog.processors.add_log_level,          # adds "level"
        structlog.processors.TimeStamper(fmt="iso"),  # adds "timestamp"
        structlog.processors.JSONRenderer(),          # render as JSON
    ],
)

log = structlog.get_logger()

MODEL_CALLS = Counter("rag_model_calls_total", "Model API calls", ["model"])
MODEL_TOKENS = Counter("rag_model_tokens_total", "Tokens consumed", ["model"])
MODEL_ERRORS = Counter("rag_model_errors_total", "Model call errors")
CACHE_HITS = Counter("rag_cache_hits_total", "LLM cache hits")
MODEL_LATENCY = Histogram(
    "rag_model_latency_seconds", "Model call latency (seconds)", ["model"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16),   # tuned for LLM latencies, not the default web buckets
)
GUARDRAIL_BLOCKS = Counter("rag_guardrail_blocks_total", "Requests blocked by input guardrail", ["category"])



def _percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = min(int(len(s) * p / 100), len(s) - 1)
    return round(s[k], 1)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.latencies_ms: list[float] = []
        self.tokens_total = 0
        self.chat_calls = 0
        self.cache_hits = 0
        self.embed_calls = 0
        self.errors = 0

    def record_llm_call(self, latency_ms, tokens):
        with self._lock:
            self.chat_calls += 1
            self.latencies_ms.append(latency_ms)
            self.tokens_total += tokens

    def record_embed(self, latency_ms, tokens):
        with self._lock:
            self.embed_calls += 1
            self.latencies_ms.append(latency_ms)
            self.tokens_total += tokens

    def record_cache_hit(self):
        with self._lock:
            self.cache_hits += 1

    def record_error(self):
        with self._lock:
            self.errors += 1

    def summary(self):
        with self._lock:
            served = self.cache_hits + self.chat_calls
            return {
                "chat_calls": self.chat_calls,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": round(self.cache_hits / served, 3) if served else 0.0,
                "embed_calls": self.embed_calls,
                "tokens_total": self.tokens_total,
                "est_cost_usd": round(self.tokens_total / 1_000_000 * 0.30, 4),  # rough blended $/1M
                "errors": self.errors,
                "latency_ms": {
                    "p50": _percentile(self.latencies_ms, 50),
                    "p95": _percentile(self.latencies_ms, 95),
                    "p99": _percentile(self.latencies_ms, 99),
                },
            }


metrics = Metrics()

