import json

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


from rag.agents.essay import write_essay
from rag.agents.qa import answer_question
from rag.agents.summary import summarize_document
from rag.generation.answer import generate_answer_stream
from rag.retrieval.search import retrieve
from rag.observability import metrics

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

@app.post("/query/stream")
def query_stream(request: QueryRequest):
    chunks = retrieve(request.question, top_k=request.top_k)

    def event_stream():
        for token in generate_answer_stream(request.question, chunks):
            yield f"data: {json.dumps(token)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/metrics/summary")
def metrics_summary():
    return metrics.summary()