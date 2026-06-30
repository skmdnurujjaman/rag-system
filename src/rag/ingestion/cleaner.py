import re


def clean_text(text: str) -> str:
    """Remove PDF-extraction noise so chunks/embeddings carry real signal."""
    # 1. Strip leading line numbers: "39   - foo" -> "- foo"
    text = re.sub(r"(?m)^\s*\d+\s+", "", text)
    # 2. Drop ASCII-art separator lines (===, ---, ___, box-drawing)
    text = re.sub(r"(?m)^[=\-_─━]{3,}\s*$", "", text)
    # 3. Non-breaking spaces -> normal spaces
    text = text.replace("\xa0", " ")
    # 4. Collapse 3+ newlines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
