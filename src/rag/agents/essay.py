from rag.gateway.llm import chat
from rag.retrieval.search import retrieve
from rag.security.prompting import UNTRUSTED_CLAUSE, fence

ESSAY_SYSTEM = (
    "You are an essay writer. Write a well-structured, coherent essay on the given topic "
    "using ONLY the provided context as source material — an introduction, a few body "
    "paragraphs, and a conclusion. Do not invent facts beyond the context; where the context "
    "is thin, stay general rather than fabricating specifics."
)

async def write_essay(topic: str, top_k: int = 8, max_words: int = 500) -> dict:
    chunks = await retrieve(topic, top_k=top_k)
    context = "\n\n".join(f"[{i + 1}] {c['content']}" for i, c in enumerate(chunks))
    marker, fenced = fence(context)
    system = ESSAY_SYSTEM + UNTRUSTED_CLAUSE.format(marker=marker)
    user_message = (
        f"Topic: {topic}\n\nSource material fenced by {marker}:\n\n{fenced}\n\n"
        f"Write an essay of about {max_words} words on the topic, grounded only in the fenced context."
    )
    essay = await chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user_message}],
        temperature=0.5, max_tokens=1500,
    )
    return {"essay": essay, "sources": chunks}