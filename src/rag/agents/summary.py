from rag.documents import get_document_text
from rag.gateway.llm import chat
from rag.security.prompting import UNTRUSTED_CLAUSE, fence

SUMMARY_SYSTEM = (
    "You write clear, faithful summaries. Summarize ONLY what is in the document — "
    "do not add outside information or opinions."
)


async def summarize_document(document_id: int, max_words: int = 200) -> str:
    text = await get_document_text(document_id)
    if not text:
        raise ValueError(f"No content found for document_id={document_id}")
    marker, fenced = fence(text)
    system = SUMMARY_SYSTEM + UNTRUSTED_CLAUSE.format(marker=marker)
    user_message = f"Summarize the document fenced by {marker} in about {max_words} words.\n\n{fenced}"
    return await chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user_message}],
        temperature=0, max_tokens=512,
    )