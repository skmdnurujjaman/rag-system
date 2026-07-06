import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from rag.agents.essay import write_essay
from rag.agents.qa import answer_question
from rag.agents.summary import summarize_document
from rag.db.pool import pool
from rag.generation.answer import generate_answer_stream
from rag.observability import GUARDRAIL_BLOCKS, log, metrics
from rag.retrieval.search import retrieve
from rag.security.guardrails import check_input
from rag.security.output_guard import guard_output

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
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.open()        # startup: open the async pool (replaces the old open=True)
    yield
    await pool.close()       # shutdown: drain and close connections

app = FastAPI(title="Agentic RAG", lifespan=lifespan)   # was: FastAPI(title="Agentic RAG")

@app.post("/essay", response_model=EssayResponse)
async def essay(request: EssayRequest) -> EssayResponse:
    guard = await check_input(request.topic)
    if not guard.allowed:
        log.warning("guardrail.blocked", category=guard.category, reason=guard.reason)
        GUARDRAIL_BLOCKS.labels(guard.category).inc()
        raise HTTPException(status_code=400, detail="Request blocked by input guardrail.")
    result = await write_essay(request.topic, top_k=request.top_k, max_words=request.max_words)
    out = await guard_output(result["essay"])
    if out.flagged:
        return EssayResponse(answer="I can't provide that response.", sources=[])
    return EssayResponse(essay=out.text, sources=result["sources"])

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    try:
        summary = await summarize_document(request.document_id, max_words=request.max_words)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    out = await guard_output(summary)
    if out.flagged:
        return SummarizeResponse(summary="I can't provide that response.")
    return SummarizeResponse(summary=out.text)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    guard = await check_input(request.question)
    if not guard.allowed:
        log.warning("guardrail.blocked", category=guard.category, reason=guard.reason)
        GUARDRAIL_BLOCKS.labels(guard.category).inc()
        raise HTTPException(status_code=400, detail="Request blocked by input guardrail.")
    result = await answer_question(request.question, top_k=request.top_k)
    out = await guard_output(result["answer"])
    if out.flagged:
        return QueryResponse(answer="I can't provide that response.", sources=[])
    return QueryResponse(answer=out.text, sources=result["chunks"])

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    chunks = await retrieve(request.question, top_k=request.top_k)

    async def event_stream():
        async for token in generate_answer_stream(request.question, chunks):
            yield f"data: {json.dumps(token)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/metrics/summary")
def metrics_summary():
    return metrics.summary()

@app.post("/alert-webhook")
async def alert_webhook(payload: dict):
    for a in payload.get("alerts", []):
        log.warning("alert.received",
                    name=a["labels"].get("alertname"),
                    severity=a["labels"].get("severity"),
                    status=a["status"],
                    summary=a["annotations"].get("summary"))
    return {"ok": True}