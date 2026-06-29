from rag.generation.answer import generate_answer
from rag.retrieval.search import search


def answer_question(question: str, top_k: int = 5) -> dict:
    """The Q&A agent: retrieve relevant chunks, then generate a grounded answer."""
    chunks = search(question, top_k=top_k)
    answer = generate_answer(question, chunks)
    return {"answer": answer, "chunks": chunks}
