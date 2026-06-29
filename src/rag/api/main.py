from fastapi import FastAPI
from pydantic import BaseModel

from rag.agents.qa import answer_question

app = FastAPI(title="Agentic RAG")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    id: int
    document_id: int
    content: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = answer_question(request.question, top_k=request.top_k)
    return QueryResponse(answer=result["answer"], sources=result["chunks"])
