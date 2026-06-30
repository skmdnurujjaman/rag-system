from openai import OpenAI

from rag.config import settings

client = OpenAI(api_key=settings.openai_api_key)
QUERY_MODEL = "gpt-4o-mini"


def multi_query(query: str, n: int = 3) -> list[str]:
    """Return the original query plus n LLM paraphrases (varied wording, same intent)."""
    prompt = (
        f"Rewrite the question below in {n} different ways, using varied wording and synonyms "
        f"while keeping the exact same meaning. Output one rewrite per line, no numbering.\n\n"
        f"Question: {query}"
    )
    response = client.chat.completions.create(
        model=QUERY_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    rewrites = [ln.strip() for ln in response.choices[0].message.content.splitlines() if ln.strip()]
    return [query, *rewrites]

def hyde(query: str) -> str:
    """Generate a hypothetical answer to embed instead of the query (HyDE)."""
    prompt = (
        "Write a short, factual paragraph that directly answers the question below, "
        "as if it were an excerpt from a technical study-notes document. "
        "Do not hedge or say you're unsure — just write a plausible answer.\n\n"
        f"Question: {query}"
    )
    response = client.chat.completions.create(
        model=QUERY_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

def decompose(query: str) -> list[str]:
    """Break a complex question into simpler standalone sub-questions."""
    prompt = (
        "If the question below contains multiple distinct sub-questions, break it into "
        "2-4 simpler standalone sub-questions, one per line, no numbering. "
        "If it is already a single simple question, return it unchanged.\n\n"
        f"Question: {query}"
    )
    response = client.chat.completions.create(
        model=QUERY_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return [ln.strip() for ln in response.choices[0].message.content.splitlines() if ln.strip()]
