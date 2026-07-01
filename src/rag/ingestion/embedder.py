from rag.gateway.llm import embed


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embed(texts)