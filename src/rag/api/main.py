import json
from contextlib import asynccontextmanager
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job
from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from rag.agents.essay import write_essay
from rag.agents.qa import answer_question
from rag.agents.summary import summarize_document
from rag.auth import require_tenant
from rag.config import settings
from rag.db.pool import pool
from rag.generation.answer import generate_answer_stream
from rag.observability import GUARDRAIL_BLOCKS, log, metrics
from rag.ratelimit import check_rate_limit
from rag.retrieval.search import retrieve
from rag.security.guardrails import check_input
from rag.security.output_guard import guard_output

RATE_LIMIT = 5        # requests…
RATE_WINDOW = 60       # …per 60s per client

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    
class IngestResponse(BaseModel):
    job_id: str
    status: str = "queued"
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.open()        # startup: open the async pool (replaces the old open=True)
    app.state.arq = await create_pool(RedisSettings.from_dsn(settings.redis_url))   # ← add
    yield
    await app.state.arq.close()                                                     # ← add
    await pool.close()       # shutdown: drain and close connections

app = FastAPI(title="Agentic RAG", lifespan=lifespan)   # was: FastAPI(title="Agentic RAG")

async def rate_limit(tenant_id: int = Depends(require_tenant)):
    remaining = await check_rate_limit(f"tenant:{tenant_id}", RATE_LIMIT, RATE_WINDOW)
    if remaining < 0:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.",
                            headers={"Retry-After": str(RATE_WINDOW)})
        
@app.post("/essay", response_model=EssayResponse, dependencies=[Depends(rate_limit)])
async def essay(request: EssayRequest, tenant_id: int = Depends(require_tenant)) -> EssayResponse:
    guard = await check_input(request.topic)
    if not guard.allowed:
        log.warning("guardrail.blocked", category=guard.category, reason=guard.reason)
        GUARDRAIL_BLOCKS.labels(guard.category).inc()
        raise HTTPException(status_code=400, detail="Request blocked by input guardrail.")
    result = await write_essay(request.topic, top_k=request.top_k, max_words=request.max_words, tenant_id=tenant_id)
    out = await guard_output(result["essay"])
    if out.flagged:
        return EssayResponse(answer="I can't provide that response.", sources=[])
    return EssayResponse(essay=out.text, sources=result["sources"])

@app.post("/summarize", response_model=SummarizeResponse, dependencies=[Depends(rate_limit)])
async def summarize(request: SummarizeRequest, tenant_id: int = Depends(require_tenant)) -> SummarizeResponse:
    try:
        summary = await summarize_document(request.document_id, max_words=request.max_words, tenant_id=tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    out = await guard_output(summary)
    if out.flagged:
        return SummarizeResponse(summary="I can't provide that response.")
    return SummarizeResponse(summary=out.text)

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
async def query(request: QueryRequest, tenant_id: int = Depends(require_tenant)) -> QueryResponse:
    guard = await check_input(request.question)
    if not guard.allowed:
        log.warning("guardrail.blocked", category=guard.category, reason=guard.reason)
        GUARDRAIL_BLOCKS.labels(guard.category).inc()
        raise HTTPException(status_code=400, detail="Request blocked by input guardrail.")
    result = await answer_question(request.question, top_k=request.top_k, tenant_id=tenant_id)
    out = await guard_output(result["answer"])
    if out.flagged:
        return QueryResponse(answer="I can't provide that response.", sources=[])
    return QueryResponse(answer=out.text, sources=result["chunks"])

@app.post("/query/stream")
async def query_stream(request: QueryRequest, tenant_id: int = Depends(require_tenant)):
    chunks = await retrieve(request.question, top_k=request.top_k, tenant_id=tenant_id)

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

@app.post("/documents", response_model=IngestResponse, status_code=202)
async def upload_document(request: Request, file: UploadFile, tenant_id: int = Depends(require_tenant)) -> IngestResponse:
    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(await file.read())                 # save the upload locally
    job = await request.app.state.arq.enqueue_job("ingest_document", str(dest), tenant_id)
    return IngestResponse(job_id=job.job_id)

@app.get("/documents/{job_id}")
async def job_status(job_id: str, request: Request):
    job = Job(job_id, request.app.state.arq)
    status = await job.status()                          # queued | in_progress | complete | not_found
    result = await job.result(timeout=0) if status == "complete" else None
    return {"job_id": job_id, "status": status, "document_id": result}