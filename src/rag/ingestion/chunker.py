def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping fixed-size character chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks
