import re


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Structure-aware chunking: group whole paragraphs up to chunk_size,
    never splitting mid-paragraph (only an oversized paragraph is hard-split)."""
    # Paragraphs = blocks separated by one or more blank (or whitespace-only) lines.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            # Oversized paragraph: flush what we have, then hard-split it (with overlap).
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), chunk_size - overlap):
                chunks.append(para[i:i + chunk_size])
            continue

        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current:
        chunks.append(current)
    return chunks
