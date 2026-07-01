from rag.gateway.llm import chat

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer is not in the context, say you don't know — do not use outside knowledge. "
    "Cite the sources you used by their number, like [1] or [2]."
)


def generate_answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{i + 1}] {c['content']}" for i, c in enumerate(chunks))
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    return chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_message}],
        temperature=0, max_tokens=1024,
    )

