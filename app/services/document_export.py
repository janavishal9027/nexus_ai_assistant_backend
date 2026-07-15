"""Document export (chat-module A.4). Turns an assistant answer (markdown) into a
downloadable file in the requested format:

    Markdown · Text · CSV · PDF · Word (.docx) · Excel (.xlsx) ·
    PowerPoint (.pptx) · zip (code-block archive)

``export_document(content, fmt, title) -> (bytes, mime_type, filename)``. A small
block parser extracts headings / paragraphs / lists / tables / code fences so each
generator can lay the content out sensibly.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from typing import Optional

_FENCE_RE = re.compile(r"^```(\w*)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")

_LANG_EXT = {
    "python": ".py", "py": ".py", "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts", "java": ".java", "c": ".c", "cpp": ".cpp",
    "c++": ".cpp", "csharp": ".cs", "cs": ".cs", "go": ".go", "rust": ".rs",
    "ruby": ".rb", "php": ".php", "swift": ".swift", "kotlin": ".kt",
    "dart": ".dart", "sql": ".sql", "bash": ".sh", "sh": ".sh", "shell": ".sh",
    "html": ".html", "css": ".css", "json": ".json", "yaml": ".yaml",
    "yml": ".yml", "xml": ".xml", "markdown": ".md", "md": ".md",
}


# ─── Lightweight markdown block parser ──────────────────────────────────────

def _parse_blocks(md: str) -> list[dict]:
    lines = (md or "").split("\n")
    n = len(lines)
    blocks: list[dict] = []
    i = 0
    while i < n:
        line = lines[i]
        fence = _FENCE_RE.match(line.strip())
        if fence:
            lang = fence.group(1)
            j, code = i + 1, []
            while j < n and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            blocks.append({"type": "code", "lang": lang, "text": "\n".join(code)})
            i = j + 1
            continue
        # table: a "| … |" line followed by a separator row
        if "|" in line and i + 1 < n and _SEP_RE.match(lines[i + 1]) and "-" in lines[i + 1]:
            k, tbl = i, []
            while k < n and "|" in lines[k]:
                tbl.append(lines[k])
                k += 1
            rows = _parse_table(tbl)
            if rows:
                blocks.append({"type": "table", "rows": rows})
                i = k
                continue
        hm = _HEADING_RE.match(line)
        if hm:
            blocks.append({"type": "heading", "level": len(hm.group(1)), "text": hm.group(2).strip()})
            i += 1
            continue
        if _LIST_RE.match(line):
            items = []
            while i < n and _LIST_RE.match(lines[i]):
                items.append(_LIST_RE.sub("", lines[i]).strip())
                i += 1
            blocks.append({"type": "list", "items": items})
            continue
        if not line.strip():
            i += 1
            continue
        para = []
        while (i < n and lines[i].strip()
               and not _FENCE_RE.match(lines[i].strip())
               and not _HEADING_RE.match(lines[i])
               and not _LIST_RE.match(lines[i])):
            para.append(lines[i].strip())
            i += 1
        blocks.append({"type": "para", "text": " ".join(para)})
    return blocks


def _parse_table(tbl_lines: list[str]) -> list[list[str]]:
    rows = []
    for l in tbl_lines:
        if _SEP_RE.match(l) and "-" in l:
            continue
        rows.append([c.strip() for c in l.strip().strip("|").split("|")])
    return rows


def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1 (\2)", text)
    return text.strip()


def _safe_title(title: str) -> str:
    title = re.sub(r"[^\w\-. ]+", "", (title or "document")).strip() or "document"
    return title[:60]


def _latin(s: str) -> str:
    return s.encode("latin-1", "replace").decode("latin-1")


# ─── Format generators ──────────────────────────────────────────────────────

def _to_markdown(content: str, title: str):
    return content.encode("utf-8"), "text/markdown", f"{title}.md"


def _to_text(content: str, title: str):
    out = []
    for b in _parse_blocks(content):
        t = b["type"]
        if t == "heading":
            out += [_strip_md(b["text"]).upper(), ""]
        elif t == "para":
            out += [_strip_md(b["text"]), ""]
        elif t == "list":
            out += ["  • " + _strip_md(x) for x in b["items"]] + [""]
        elif t == "code":
            out += [b["text"], ""]
        elif t == "table":
            out += ["   ".join(_strip_md(c) for c in r) for r in b["rows"]] + [""]
    return ("\n".join(out).strip() or content).encode("utf-8"), "text/plain", f"{title}.txt"


def _to_csv(content: str, title: str):
    blocks = _parse_blocks(content)
    table = next((b["rows"] for b in blocks if b["type"] == "table"), None)
    buf = io.StringIO()
    w = csv.writer(buf)
    if table:
        for r in table:
            w.writerow([_strip_md(c) for c in r])
    else:
        for b in blocks:
            if b["type"] == "para":
                w.writerow([_strip_md(b["text"])])
            elif b["type"] == "list":
                for x in b["items"]:
                    w.writerow([_strip_md(x)])
            elif b["type"] == "heading":
                w.writerow([_strip_md(b["text"])])
    return buf.getvalue().encode("utf-8"), "text/csv", f"{title}.csv"


def _to_docx(content: str, title: str):
    import docx
    from docx.shared import Pt
    d = docx.Document()
    d.add_heading(title, 0)
    for b in _parse_blocks(content):
        t = b["type"]
        if t == "heading":
            d.add_heading(_strip_md(b["text"]), min(b["level"], 4))
        elif t == "para":
            d.add_paragraph(_strip_md(b["text"]))
        elif t == "list":
            for x in b["items"]:
                d.add_paragraph(_strip_md(x), style="List Bullet")
        elif t == "code":
            p = d.add_paragraph()
            run = p.add_run(b["text"])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif t == "table" and b["rows"]:
            cols = max(len(r) for r in b["rows"])
            table = d.add_table(rows=0, cols=cols)
            table.style = "Light Grid Accent 1"
            for r in b["rows"]:
                cells = table.add_row().cells
                for ci, val in enumerate(r[:cols]):
                    cells[ci].text = _strip_md(val)
    buf = io.BytesIO()
    d.save(buf)
    return (buf.getvalue(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            f"{title}.docx")


def _pdf_write(pdf, h: float, text: str) -> None:
    """Write a block to the PDF, hard-wrapping very long unbreakable tokens (so
    fpdf can always place them) and never letting one line break the export."""
    for raw in (text or "").split("\n"):
        line = _latin(raw)
        # Insert a break opportunity inside tokens longer than ~90 chars.
        line = re.sub(r"(\S{90})", r"\1 ", line)
        try:
            pdf.multi_cell(0, h, line if line else " ")
        except Exception:
            continue


def _to_pdf(content: str, title: str):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    _pdf_write(pdf, 9, title)
    pdf.ln(2)
    for b in _parse_blocks(content):
        t = b["type"]
        if t == "heading":
            pdf.set_font("Helvetica", "B", max(11, 17 - b["level"]))
            _pdf_write(pdf, 7, _strip_md(b["text"]))
            pdf.ln(1)
        elif t == "para":
            pdf.set_font("Helvetica", "", 11)
            _pdf_write(pdf, 6, _strip_md(b["text"]))
            pdf.ln(1)
        elif t == "list":
            pdf.set_font("Helvetica", "", 11)
            for x in b["items"]:
                _pdf_write(pdf, 6, "-  " + _strip_md(x))
            pdf.ln(1)
        elif t == "code":
            pdf.set_font("Courier", "", 8)
            _pdf_write(pdf, 4.5, b["text"])
            pdf.ln(1)
        elif t == "table" and b["rows"]:
            pdf.set_font("Helvetica", "", 9)
            for r in b["rows"]:
                _pdf_write(pdf, 5, " | ".join(_strip_md(c) for c in r))
            pdf.ln(1)
    return bytes(pdf.output()), "application/pdf", f"{title}.pdf"


def _to_xlsx(content: str, title: str):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    blocks = _parse_blocks(content)
    tables = [b["rows"] for b in blocks if b["type"] == "table"]
    if tables:
        for ti, rows in enumerate(tables):
            sheet = ws if ti == 0 else wb.create_sheet(f"Table {ti + 1}")
            for r in rows:
                sheet.append([_strip_md(c) for c in r])
    else:
        for b in blocks:
            if b["type"] in ("para", "heading"):
                ws.append([_strip_md(b["text"])])
            elif b["type"] == "list":
                for x in b["items"]:
                    ws.append([_strip_md(x)])
    buf = io.BytesIO()
    wb.save(buf)
    return (buf.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            f"{title}.xlsx")


def _to_pptx(content: str, title: str):
    from pptx import Presentation
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[0]).shapes.title.text = title

    state = {"title": "Overview", "bullets": []}

    def flush():
        if not state["bullets"]:
            return
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = state["title"][:80]
        tf = slide.placeholders[1].text_frame
        tf.clear()
        for bi, bl in enumerate(state["bullets"][:10]):
            p = tf.paragraphs[0] if bi == 0 else tf.add_paragraph()
            p.text = bl[:200]
        state["bullets"] = []

    for b in _parse_blocks(content):
        if b["type"] == "heading":
            flush()
            state["title"] = _strip_md(b["text"])
        elif b["type"] == "para":
            state["bullets"].append(_strip_md(b["text"]))
        elif b["type"] == "list":
            state["bullets"].extend(_strip_md(x) for x in b["items"])
    flush()
    buf = io.BytesIO()
    prs.save(buf)
    return (buf.getvalue(),
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            f"{title}.pptx")


def _to_zip(content: str, title: str):
    blocks = _parse_blocks(content)
    codes = [b for b in blocks if b["type"] == "code" and b["text"].strip()]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", content or "")
        seen: dict[str, int] = {}
        for b in codes:
            ext = _LANG_EXT.get((b["lang"] or "").lower(), ".txt")
            base = (b["lang"] or "file").lower()
            seen[base] = seen.get(base, 0) + 1
            z.writestr(f"project/{base}_{seen[base]}{ext}", b["text"])
    return buf.getvalue(), "application/zip", f"{title}.zip"


_GENERATORS = {
    "md": _to_markdown, "markdown": _to_markdown,
    "txt": _to_text, "text": _to_text,
    "csv": _to_csv,
    "pdf": _to_pdf,
    "docx": _to_docx, "word": _to_docx,
    "xlsx": _to_xlsx, "excel": _to_xlsx,
    "pptx": _to_pptx, "powerpoint": _to_pptx,
    "zip": _to_zip,
}

SUPPORTED_FORMATS = ["markdown", "word", "pdf", "excel", "csv", "text", "powerpoint", "zip"]


def export_document(content: str, fmt: str, title: str = "document"):
    """Return (bytes, mime_type, filename) for the requested format."""
    gen = _GENERATORS.get((fmt or "md").lower().strip())
    if gen is None:
        raise ValueError(f"Unsupported export format: {fmt}")
    return gen(content or "", _safe_title(title))
