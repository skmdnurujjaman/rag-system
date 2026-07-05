import sys

from evals.run_essay_eval import evaluate as run_essay_eval
from evals.run_generation_eval import evaluate as run_generation_eval
from evals.run_retrieval_eval import evaluate as run_retrieval_eval
from evals.run_security_eval import evaluate as run_security_eval
from evals.run_summary_eval import evaluate as run_summary_eval

# Floors = current baseline (or just below, to tolerate LLM-judge noise).
# Raise these as you improve — that's the ratchet.
MIN_THRESHOLDS = {
    "recall_at_k": 0.80,
    "faithfulness": 4.5,          # Q&A generation faithfulness
    "correctness": 4.5,           # Q&A correctness
    "summary_faithfulness": 4.5,
    "essay_faithfulness": 4.5,
}

MAX_THRESHOLDS = {          # value <= ceiling
    "attack_success_rate": 0.0,     # ZERO attacks may pass
    "false_positive_rate": 0.20,    # tolerate a little over-blocking
}

def main() -> None:
    retrieval = run_retrieval_eval()
    generation = run_generation_eval()
    actual = {
        "recall_at_k": retrieval["recall_at_k"],
        "faithfulness": generation["faithfulness"],
        "correctness": generation["correctness"],
        "summary_faithfulness": run_summary_eval()["summary_faithfulness"],
        "essay_faithfulness": run_essay_eval()["essay_faithfulness"],
        **run_security_eval()
    }


    print("\n=== EVAL GATE ===")
    failed = False
    for name, floor in MIN_THRESHOLDS.items():
        value = actual[name]
        ok = value >= floor
        failed = failed or not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.3f}  (min {floor})")
    for name, ceil in MAX_THRESHOLDS.items():
        ok = actual[name] <= ceil
        failed |= not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {actual[name]:.3f}  (max {ceil})")

    if failed:
        print("GATE FAILED — quality regressed below threshold.")
        sys.exit(1)
    print("GATE PASSED.")


if __name__ == "__main__":
    main()
