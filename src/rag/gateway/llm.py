import logging

from openai import OpenAI

from rag.config import settings

logger = logging.getLogger(__name__)
_client = OpenAI(api_key=settings.openai_api_key)

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"


def chat(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0,
         max_tokens: int = 1024, **kwargs) -> str:
    """Single entry point for all chat-completion calls."""
    logger.info("gateway.chat model=%s messages=%d", model, len(messages))
    response = _client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, **kwargs,
    )
    return response.choices[0].message.content


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Single entry point for all embedding calls."""
    logger.info("gateway.embed model=%s texts=%d", model, len(texts))
    response = _client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
