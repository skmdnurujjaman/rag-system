import asyncio
import logging
import operator
import uuid
from typing import Annotated, Optional, TypedDict

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from rag.agents.essay import write_essay
from rag.agents.qa import answer_question
from rag.agents.summary import summarize_document
from rag.config import settings
from rag.gateway.llm import chat, start_budget
from rag.retrieval.query_transform import decompose
from rag.security.guardrails import check_input

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    request: str
    document_id: Optional[int]
    intent: str
    subquestions: list[str]
    subanswers: Annotated[list[str], operator.add]
    result: dict

async def _classify(request: str) -> str:
    prompt = (
        "Classify the request into exactly one word: qa, summary, or essay.\n"
        "- qa: answer a question from the documents\n"
        "- summary: summarize a document\n"
        "- essay: write an essay on a topic\n"
        f"\nRequest: {request}\n\nAnswer with only one word:"
    )
    intent = (await chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=5)).strip().lower()
    return intent if intent in {"qa", "summary", "essay"} else "qa"

async def router_node(state: AgentState) -> dict:
    intent = await _classify(state["request"])
    logger.info("router intent=%s request=%r", intent, state["request"][:60])
    return {"intent": intent}

async def summary_node(state: AgentState) -> dict:
    return {"result": {"summary": await summarize_document(state["document_id"])}}


async def essay_node(state: AgentState) -> dict:
    return {"result": await write_essay(state["request"])}


def _route(state: AgentState) -> str:
    return state["intent"]

# --- Q&A with conditional decomposition ---

async def qa_plan_node(state: AgentState) -> dict:
    subs = await decompose(state["request"])           # [question] if simple; multiple if multi-hop
    logger.info("qa_plan subquestions=%d", len(subs))
    return {"subquestions": subs}

async def qa_answer_node(state: AgentState) -> dict:
    sub = state["subquestions"][0]
    logger.info("qa_answer sub=%r", sub[:50])
    answer = (await answer_question(sub))["answer"]
    return {"subquestions": state["subquestions"][1:], "subanswers": [answer]}

def qa_should_continue(state: AgentState) -> str:
    return "qa_answer" if state["subquestions"] else "qa_synthesize"

async def qa_synthesize_node(state: AgentState) -> dict:
    answers = state["subanswers"]
    if len(answers) == 1:
        return {"result": {"answer": answers[0]}}
    joined = "\n\n".join(f"- {a}" for a in answers)
    final = await chat(
        [
            {"role": "system", "content":
             "Combine the partial answers into one coherent answer to the user's original "
             "question. Use only the information provided."},
            {"role": "user", "content":
             f"Original question: {state['request']}\n\nPartial answers:\n{joined}"},
        ],
        temperature=0, max_tokens=800,
    )
    return {"result": {"answer": final}}

def build_graph(checkpointer):
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("summary", summary_node)
    graph.add_node("essay", essay_node)
    graph.add_node("qa_plan", qa_plan_node)
    graph.add_node("qa_answer", qa_answer_node)
    graph.add_node("qa_synthesize", qa_synthesize_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", _route,
                                {"qa": "qa_plan", "summary": "summary", "essay": "essay"})
    graph.add_edge("qa_plan", "qa_answer")
    graph.add_conditional_edges("qa_answer", qa_should_continue,
                                {"qa_answer": "qa_answer", "qa_synthesize": "qa_synthesize"})
    graph.add_edge("qa_synthesize", END)
    graph.add_edge("summary", END)
    graph.add_edge("essay", END)
    
    return graph.compile(checkpointer=checkpointer)


_agent_graph = None
_init_lock = asyncio.Lock()

async def _get_graph():
    """Create the async checkpointer + graph once, on first use (can't await at import)."""
    global _agent_graph
    if _agent_graph is None:
        async with _init_lock:                    # double-checked lock: concurrent first-calls don't race
            if _agent_graph is None:
                conn = await AsyncConnection.connect(
                    settings.database_url, autocommit=True, row_factory=dict_row,
                )
                checkpointer = AsyncPostgresSaver(conn)
                await checkpointer.setup()         # creates checkpoint tables if missing
                _agent_graph = build_graph(checkpointer)
    return _agent_graph

async def run_agent(request: str, document_id: Optional[int] = None,
              token_budget: int = 0, recursion_limit: int = 15, thread_id: Optional[str] = None) -> dict:
    """Run the agent graph with a per-request token budget and a step (loop) limit."""
    guard = await check_input(request)
    if not guard.allowed:
        logger.warning("agent blocked by guardrail category=%s", guard.category)
        raise ValueError(f"Request blocked by input guardrail: {guard.category}")
    graph = await _get_graph()
    thread_id = thread_id or str(uuid.uuid4())
    start_budget(token_budget)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    return await graph.ainvoke(
        {"request": request, "document_id": document_id, "subquestions": [], "subanswers": []},
        config,
    )