from rag.gateway.llm import embed


async def embed_texts(texts: list[str]) -> list[list[float]]:
    return await embed(texts)