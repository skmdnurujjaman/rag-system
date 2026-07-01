from fastapi import FastAPI
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException

from rag.agents.qa import answer_question
from rag.agents.summary import summarize_document
from rag.agents.essay import write_essay

app = FastAPI(title="Agentic RAG")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    id: int
    document_id: int
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    
class SummarizeRequest(BaseModel):
    document_id: int
    max_words: int = 200


class SummarizeResponse(BaseModel):
    summary: str

class EssayRequest(BaseModel):
    topic: str
    top_k: int = 8
    max_words: int = 500

class EssayResponse(BaseModel):
    essay: str
    sources: list[Source]
    
@app.post("/essay", response_model=EssayResponse)
def essay(request: EssayRequest) -> EssayResponse:
    result = write_essay(request.topic, top_k=request.top_k, max_words=request.max_words)
    return EssayResponse(essay=result["essay"], sources=result["sources"])

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    try:
        summary = summarize_document(request.document_id, max_words=request.max_words)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return SummarizeResponse(summary=summary)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = answer_question(request.question, top_k=request.top_k)
    return QueryResponse(answer=result["answer"], sources=result["chunks"])
