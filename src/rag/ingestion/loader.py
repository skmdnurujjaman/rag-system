from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """Extract all text from a PDF file as a single string."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
