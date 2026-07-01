from rag.gateway.llm import chat
from rag.retrieval.search import retrieve

ESSAY_SYSTEM = (
    "You are an essay writer. Write a well-structured, coherent essay on the given topic "
    "using ONLY the provided context as source material — an introduction, a few body "
    "paragraphs, and a conclusion. Do not invent facts beyond the context; where the context "
    "is thin, stay general rather than fabricating specifics."
)

def write_essay(topic: str, top_k: int = 8, max_words: int = 500) -> dict:
    chunks = retrieve(topic, top_k=top_k)
    context = "\n\n".join(f"[{i + 1}] {c['content']}" for i, c in enumerate(chunks))
    user_message = (
        f"Topic: {topic}\n\nContext (source material):\n{context}\n\n"
        f"Write an essay of about {max_words} words on the topic, grounded in the context above."
    )
    essay = chat(
        [{"role": "system", "content": ESSAY_SYSTEM},
         {"role": "user", "content": user_message}],
        temperature=0.5, max_tokens=1500,
    )
    return {"essay": essay, "sources": chunks}
