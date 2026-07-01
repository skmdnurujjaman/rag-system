from rag.gateway.llm import chat
from rag.gateway.llm import chat, chat_stream
from rag.gateway.llm import chat, chat_stream, route_model

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer is not in the context, say you don't know — do not use outside knowledge. "
    "Cite the sources you used by their number, like [1] or [2]."
)

def _build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{i + 1}] {c['content']}" for i, c in enumerate(chunks))
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

def generate_answer(question: str, chunks: list[dict]) -> str:
    return chat(_build_messages(question, chunks), model=route_model(question),
                temperature=0, max_tokens=1024)


def generate_answer_stream(question: str, chunks: list[dict]):
    yield from chat_stream(_build_messages(question, chunks), model=route_model(question),
                           temperature=0, max_tokens=1024)