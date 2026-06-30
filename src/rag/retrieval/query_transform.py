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
