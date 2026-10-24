"""PDF extraction and deterministic text chunking."""

import re
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_paragraphs(path: str | Path) -> list[str]:
    """Extract non-empty text blocks in PDF reading order."""
    paragraphs: list[str] = []
    with pymupdf.open(Path(path)) as document:
        for page in document:
            blocks = sorted(page.get_text("blocks"), key=lambda block: (block[1], block[0]))
            paragraphs.extend(
                normalized for block in blocks if (normalized := _normalize(block[4]))
            )
    if not paragraphs:
        raise ValueError(f"No text could be extracted from PDF: {path}")
    return paragraphs


def chunk_paragraphs(
    paragraphs: list[str], *, chunk_size: int = 1_000, chunk_overlap: int = 150
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [
        chunk.strip() for chunk in splitter.split_text("\n\n".join(paragraphs)) if chunk.strip()
    ]
