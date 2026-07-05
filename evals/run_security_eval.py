import json
from pathlib import Path

from rag.security.guardrails import check_input

DATASET = Path(__file__).parent / "security_dataset.json"

def evaluate() -> dict:
    cases = json.loads(DATASET.read_text())
    attacks = [c for c in cases if c["should_block"]]
    benign  = [c for c in cases if not c["should_block"]]

    breaches     = [c for c in attacks if check_input(c["input"]).allowed]        # attack got through
    false_blocks = [c for c in benign  if not check_input(c["input"]).allowed]    # benign blocked

    for c in breaches:      
        print(f"  [BREACH]         {c['input']!r}")
    for c in false_blocks:  
        print(f"  [FALSE-POSITIVE] {c['input']!r}")

    return {
        "attack_success_rate": round(len(breaches) / len(attacks), 3) if attacks else 0.0,
        "false_positive_rate": round(len(false_blocks) / len(benign), 3) if benign else 0.0,
    }

if __name__ == "__main__":
    print(evaluate())
