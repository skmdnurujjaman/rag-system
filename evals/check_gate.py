import sys

from evals.run_generation_eval import evaluate as run_generation_eval
from evals.run_retrieval_eval import evaluate as run_retrieval_eval

# Floors = current baseline (or just below, to tolerate LLM-judge noise).
# Raise these as you improve — that's the ratchet.
THRESHOLDS = {
    "recall_at_k": 0.80,
    "faithfulness": 4.5,
    "correctness": 3.0,
}


def main() -> None:
    retrieval = run_retrieval_eval()
    generation = run_generation_eval()
    actual = {
        "recall_at_k": retrieval["recall_at_k"],
        "faithfulness": generation["faithfulness"],
        "correctness": generation["correctness"],
    }

    print("\n=== EVAL GATE ===")
    failed = False
    for name, floor in THRESHOLDS.items():
        value = actual[name]
        ok = value >= floor
        failed = failed or not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.3f}  (min {floor})")

    if failed:
        print("GATE FAILED — quality regressed below threshold.")
        sys.exit(1)
    print("GATE PASSED.")


if __name__ == "__main__":
    main()
