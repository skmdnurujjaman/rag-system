import logging
import operator
import uuid
import psycopg

from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from rag.config import settings 
from typing import Annotated, Optional, TypedDict
from langgraph.graph import END, START, StateGraph

from rag.agents.essay import write_essay
from rag.agents.qa import answer_question
from rag.agents.summary import summarize_document
from rag.gateway.llm import chat, start_budget
from rag.retrieval.query_transform import decompose

logger = logging.getLogger(__name__)

_checkpointer_conn = psycopg.connect(
    settings.database_url,
    autocommit=True,
    row_factory=dict_row,
)
_checkpointer = PostgresSaver(_checkpointer_conn)
_checkpointer.setup()


class AgentState(TypedDict):
    request: str
    document_id: Optional[int]
    intent: str
    subquestions: list[str]
    subanswers: Annotated[list[str], operator.add]
    result: dict


def _classify(request: str) -> str:
    prompt = (
        "Classify the request into exactly one word: qa, summary, or essay.\n"
        "- qa: answer a question from the documents\n"
        "- summary: summarize a document\n"
        "- essay: write an essay on a topic\n"
        f"\nRequest: {request}\n\nAnswer with only one word:"
    )
    intent = chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=5).strip().lower()
    return intent if intent in {"qa", "summary", "essay"} else "qa"

def router_node(state: AgentState) -> dict:
    intent = _classify(state["request"])
    logger.info("router intent=%s request=%r", intent, state["request"][:60])
    return {"intent": intent}

def summary_node(state: AgentState) -> dict:
    return {"result": {"summary": summarize_document(state["document_id"])}}


def essay_node(state: AgentState) -> dict:
    return {"result": write_essay(state["request"])}


def _route(state: AgentState) -> str:
    return state["intent"]

# --- Q&A with conditional decomposition ---

def qa_plan_node(state: AgentState) -> dict:
    subs = decompose(state["request"])           # [question] if simple; multiple if multi-hop
    logger.info("qa_plan subquestions=%d", len(subs))
    return {"subquestions": subs}

def qa_answer_node(state: AgentState) -> dict:
    sub = state["subquestions"][0]
    logger.info("qa_answer sub=%r", sub[:50])
    answer = answer_question(sub)["answer"]
    return {"subquestions": state["subquestions"][1:], "subanswers": [answer]}

def qa_should_continue(state: AgentState) -> str:
    return "qa_answer" if state["subquestions"] else "qa_synthesize"

def qa_synthesize_node(state: AgentState) -> dict:
    answers = state["subanswers"]
    if len(answers) == 1:
        return {"result": {"answer": answers[0]}}
    joined = "\n\n".join(f"- {a}" for a in answers)
    final = chat(
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

def build_graph():
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
    
    return graph.compile(checkpointer=_checkpointer)



agent_graph = build_graph()

def run_agent(request: str, document_id: Optional[int] = None,
              token_budget: int = 0, recursion_limit: int = 15, thread_id: Optional[str] = None) -> dict:
    """Run the agent graph with a per-request token budget and a step (loop) limit."""
    thread_id = thread_id or str(uuid.uuid4())
    start_budget(token_budget)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    return agent_graph.invoke(
        {"request": request, "document_id": document_id, "subquestions": [], "subanswers": []},
        config,
    )