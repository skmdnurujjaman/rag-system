import re
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from rag.gateway.llm import moderate
from rag.observability import OUTPUT_FLAGGED, OUTPUT_PII, log

# Ordered: more specific patterns first so they win over generic ones.
_PII_PATTERNS = {
    "EMAIL":       re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "SSN":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE":       re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"),
    "IP":          re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Pin Presidio to the small model we downloaded (default hunts for en_core_web_lg and errors).
_provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
})
_analyzer = AnalyzerEngine(nlp_engine=_provider.create_engine())
_anonymizer = AnonymizerEngine()

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

def redact_pii_ner(text: str) -> tuple[str, list[str]]:
    """NER-based redaction (PERSON, LOCATION, etc.). Returns (clean_text, entity_types)."""
    results = _analyzer.analyze(text=text, language="en", entities=["PERSON", "LOCATION", "PHONE_NUMBER"])
    clean = _anonymizer.anonymize(text=text, analyzer_results=results).text
    types = sorted({r.entity_type for r in results})
    return clean, types

def check_output(text: str) -> OutputResult:
    text, pii_regex = redact_pii(text)      # layer 1: deterministic structured PII
    text, pii_ner = redact_pii_ner(text)    # layer 2: NER names/locations/etc.
    flagged, cats = moderate(text)
    pii = sorted(set(pii_regex) | set(pii_ner))
    return OutputResult(text=text, pii=pii, flagged=flagged, categories=cats)

def guard_output(text: str) -> OutputResult:
    """check_output + record metrics/logs. Caller inspects .flagged to choose its response."""
    out = check_output(text)
    for t in out.pii:
        OUTPUT_PII.labels(t).inc()
    if out.pii:
        log.warning("output.pii_redacted", types=out.pii)
    if out.flagged:
        for c in out.categories:
            OUTPUT_FLAGGED.labels(c).inc()
        log.warning("output.moderation_flagged", categories=out.categories)
    return out
