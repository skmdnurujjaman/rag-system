from rag.documents import get_document_text
from rag.gateway.llm import chat

SUMMARY_SYSTEM = (
    "You write clear, faithful summaries. Summarize ONLY what is in the document — "
    "do not add outside information or opinions."
)

def summarize_document(document_id: int, max_words: int = 200) -> str:
    text = get_document_text(document_id)
    if not text:
        raise ValueError(f"No content found for document_id={document_id}")
    user_message = f"Summarize the following document in about {max_words} words:\n\n{text}"
    return chat(
        [{"role": "system", "content": SUMMARY_SYSTEM},
         {"role": "user", "content": user_message}],
        temperature=0, max_tokens=512,
    )
