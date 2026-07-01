from rag.gateway.llm import chat


def multi_query(query: str, n: int = 3) -> list[str]:
    prompt = (
        f"Rewrite the question below in {n} different ways, using varied wording and synonyms "
        f"while keeping the exact same meaning. Output one rewrite per line, no numbering.\n\n"
        f"Question: {query}"
    )
    text = chat([{"role": "user", "content": prompt}], temperature=0)
    return [query, *[ln.strip() for ln in text.splitlines() if ln.strip()]]

def hyde(query: str) -> str:
    """Generate a hypothetical answer to embed instead of the query (HyDE)."""
    prompt = (
        "Write a short, factual paragraph that directly answers the question below, "
        "as if it were an excerpt from a technical study-notes document. "
        "Do not hedge or say you're unsure — just write a plausible answer.\n\n"
        f"Question: {query}"
    )
    text = chat([{"role": "user", "content": prompt}], temperature=0)
    return text.strip()

def decompose(query: str) -> list[str]:
    """Break a complex question into simpler standalone sub-questions."""
    prompt = (
        "If the question below contains multiple distinct sub-questions, break it into "
        "2-4 simpler standalone sub-questions, one per line, no numbering. "
        "If it is already a single simple question, return it unchanged.\n\n"
        f"Question: {query}"
    )
    text = chat([{"role": "user", "content": prompt}], temperature=0)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]
