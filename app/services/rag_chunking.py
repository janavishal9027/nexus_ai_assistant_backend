"""Document text extraction, cleaning, and chunking for RAG ingestion.

Text/markdown/code are decoded directly (stdlib). PDF uses ``pypdf`` and DOCX
uses ``python-docx`` — both pure-Python and lazily imported, so the service
still loads if they are absent (an unsupported file then fails its own job
cleanly rather than breaking the app).
"""
from __future__ import annotations

import io
import os
import re
import unicodedata

# Extensions decoded as plain UTF-8 text.
_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".mdx", ".rst", ".csv", ".tsv", ".json",
    ".log", ".yaml", ".yml", ".xml", ".html", ".htm", ".ini", ".cfg", ".toml",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".sql", ".dart", ".kt", ".swift",
}
_PDF_EXTS = {".pdf"}
_DOCX_EXTS = {".docx"}

SUPPORTED_EXTS = _TEXT_EXTS | _PDF_EXTS | _DOCX_EXTS


def is_supported(filename: str) -> bool:
    return _ext(filename) in SUPPORTED_EXTS


def _ext(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def extract_text(filename: str, content: bytes) -> str:
    """Extract raw text from an uploaded file's bytes. Raises ValueError with a
    user-facing message for unsupported or unreadable files."""
    ext = _ext(filename)
    if ext in _PDF_EXTS:
        return _extract_pdf(content)
    if ext in _DOCX_EXTS:
        return _extract_docx(content)
    if ext in _TEXT_EXTS:
        return content.decode("utf-8", errors="replace")
    # Unknown extension: accept it only if it decodes as mostly-printable text.
    text = content.decode("utf-8", errors="replace")
    if _looks_binary(text):
        raise ValueError(
            f"Unsupported file type '{ext or filename}'. Supported: PDF, DOCX, "
            f"and text/markdown/code files."
        )
    return text


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover
        raise ValueError("PDF support unavailable (pypdf not installed)") from exc
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError(
            "No extractable text in this PDF (it may be scanned images — OCR "
            "is not supported)."
        )
    return text


def _extract_docx(content: bytes) -> str:
    try:
        import docx  # python-docx
    except Exception as exc:  # pragma: no cover
        raise ValueError("DOCX support unavailable (python-docx not installed)") from exc
    try:
        document = docx.Document(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not read DOCX: {exc}") from exc
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError("No extractable text in this DOCX.")
    return text


def _looks_binary(text: str) -> bool:
    if not text:
        return True
    sample = text[:2000]
    # Many replacement chars or NULs → the bytes weren't really text.
    bad = sample.count("�") + sample.count("\x00")
    return bad > max(10, len(sample) * 0.1)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters while preserving
    paragraph structure (blank lines) so chunking can break on boundaries."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Drop control chars except newline and tab.
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    # Collapse runs of spaces/tabs; trim each line; cap consecutive blank lines.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_BREAKS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split cleaned text into overlapping chunks no larger than ``chunk_size``
    characters, preferring to cut on paragraph/sentence/word boundaries."""
    text = (text or "").strip()
    n = len(text)
    if n == 0:
        return []
    if n <= chunk_size:
        return [text]

    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            cut = _best_break(window, int(chunk_size * 0.5))
            if cut > 0:
                end = start + cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _best_break(window: str, min_index: int) -> int:
    """Index just past the best boundary in ``window`` at or after ``min_index``,
    or -1 if none — so we never cut absurdly early."""
    for pat in _BREAKS:
        idx = window.rfind(pat)
        if idx >= min_index:
            return idx + len(pat)
    return -1
