from openai import OpenAI

from rag.config import settings
from rag.documents import get_document_text

client = OpenAI(api_key=settings.openai_api_key)
SUMMARY_MODEL = "gpt-4o-mini"

SUMMARY_SYSTEM = (
    "You write clear, faithful summaries. Summarize ONLY what is in the document — "
    "do not add outside information or opinions."
)


def summarize_document(document_id: int, max_words: int = 200) -> str:
    """Summarize a whole ingested document (stuff strategy)."""
    text = get_document_text(document_id)
    if not text:
        raise ValueError(f"No content found for document_id={document_id}")

    user_message = f"Summarize the following document in about {max_words} words:\n\n{text}"
    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        max_tokens=512,
        temperature=0,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
