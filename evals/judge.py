import json

from pydantic import BaseModel

from rag.gateway.llm import chat


class Judgment(BaseModel):
    score: int
    reasoning: str

async def _judge(system_prompt: str, user_prompt: str) -> Judgment:
    content = await chat(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        temperature=0, max_tokens=512, response_format={"type": "json_object"},
    )
    return Judgment(**json.loads(content))

CORRECTNESS_SYSTEM = """You grade whether a GENERATED answer is factually correct \
compared to a reference EXPECTED answer for the same question. Judge meaning, not wording.
Score 1-5:
5 = all key facts present and correct
4 = correct but missing a minor detail
3 = partially correct; a key fact missing or a small error
2 = mostly incorrect; only a fragment is right
1 = incorrect or contradicts the expected answer
Respond ONLY as JSON: {"score": <int 1-5>, "reasoning": "<one sentence>"}"""


async def judge_correctness(question: str, expected: str, generated: str) -> Judgment:
    user = f"Question: {question}\n\nEXPECTED answer: {expected}\n\nGENERATED answer: {generated}"
    return await _judge(CORRECTNESS_SYSTEM, user)


FAITHFULNESS_SYSTEM = """You check a GENERATED answer for hallucination against CONTEXT passages.
Judge whether every claim in the answer is supported by the context.
Score 1-5:
5 = every claim is fully supported by the context
4 = supported, with one trivial unsupported aside
3 = mostly supported; one notable claim is not in the context
2 = several claims unsupported
1 = key claims contradict or are absent from the context
Note: answering "I don't know" when the context lacks the answer is FULLY FAITHFUL (score 5).
Respond ONLY as JSON: {"score": <int 1-5>, "reasoning": "<one sentence>"}"""


async def judge_faithfulness(context_chunks: list[str], generated: str) -> Judgment:
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    user = f"CONTEXT:\n{context}\n\nGENERATED answer: {generated}"
    return await _judge(FAITHFULNESS_SYSTEM, user)
