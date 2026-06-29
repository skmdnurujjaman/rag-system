from openai import OpenAI

from rag.config import settings

client = OpenAI(api_key=settings.openai_api_key)

ANSWER_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer is not in the context, say you don't know — do not use outside knowledge. "
    "Cite the sources you used by their number, like [1] or [2]."
)


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate a grounded, cited answer from retrieved chunks."""
    context = "\n\n".join(f"[{i + 1}] {chunk['content']}" for i, chunk in enumerate(chunks))
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content

