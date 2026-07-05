import re
from dataclasses import dataclass, field
from rag.gateway.llm import moderate

# Ordered: more specific patterns first so they win over generic ones.
_PII_PATTERNS = {
    "EMAIL":       re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "SSN":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE":       re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"),
    "IP":          re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

@dataclass
class OutputResult:
    text: str
    pii: list[str] = field(default_factory=list)
    flagged: bool = False
    categories: list[str] = field(default_factory=list)

def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace detected PII with [REDACTED_<TYPE>]. Returns (clean_text, types_found)."""
    found: list[str] = []
    for label, rx in _PII_PATTERNS.items():
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[REDACTED_{label}]", text)
    return text, found

def check_output(text: str) -> OutputResult:
    clean, pii = redact_pii(text)          # redact first…
    flagged, cats = moderate(clean)        # …then moderate the cleaned text
    return OutputResult(text=clean, pii=pii, flagged=flagged, categories=cats)
