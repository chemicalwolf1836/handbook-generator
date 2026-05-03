"""
PDF Processor — extracts and chunks text from uploaded PDF files.
Uses pdfplumber as primary extractor, falls back to pypdf.
"""
import re
from pathlib import Path


def extract_text_from_pdf(file_path: str) -> str:
    """Extract full text from a PDF file."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception:
        pass

    # Fallback to pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise RuntimeError(f"Could not extract text from PDF: {e}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks for RAG indexing.
    Tries to split on paragraph/sentence boundaries.
    """
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Split on paragraph boundaries first
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += ("\n\n" + para if current else para)
        else:
            if current:
                chunks.append(current.strip())
            # If a single paragraph is too long, split by sentence
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) < chunk_size:
                        sub += (" " + sent if sub else sent)
                    else:
                        if sub:
                            chunks.append(sub.strip())
                        sub = sent
                if sub:
                    current = sub
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current.strip())

    # Add overlap: prepend end of previous chunk to next chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(tail + "\n" + chunks[i])
        return overlapped

    return chunks


def get_doc_metadata(file_path: str) -> dict:
    """Extract basic metadata from a PDF."""
    meta = {"filename": Path(file_path).name, "pages": 0, "title": ""}
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        meta["pages"] = len(reader.pages)
        info = reader.metadata
        if info:
            meta["title"] = info.get("/Title", "") or Path(file_path).stem
    except Exception:
        meta["title"] = Path(file_path).stem
    return meta
