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
        # Keep the source line breaks (single newlines) rather than collapsing
        # them into one run-on line — many answers put structured content (e.g.
        # "Label: values" rows) on their own lines and rely on that layout.
        blocks.append({"type": "para", "text": "\n".join(para)})
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


def _docx_add_runs(paragraph, text: str) -> None:
    """Append text to a docx paragraph, rendering inline **bold** as bold runs
    (other markers are stripped). Keeps emphasis instead of flattening it."""
    for seg in re.split(r"(\*\*.+?\*\*)", text):
        if not seg:
            continue
        if len(seg) > 4 and seg.startswith("**") and seg.endswith("**"):
            paragraph.add_run(_strip_md(seg[2:-2])).bold = True
        else:
            paragraph.add_run(_strip_md(seg))


def _norm(s: str) -> str:
    """Normalize for comparison (lowercase alphanumerics only)."""
    return re.sub(r"[^a-z0-9]+", "", _strip_md(s or "").lower())


def _skip_title(blocks: list[dict], title: str) -> bool:
    """True when the content already opens with the title heading, so the
    generator shouldn't add a second (duplicate) title on top."""
    return bool(blocks and blocks[0]["type"] == "heading"
                and _norm(blocks[0]["text"]) == _norm(title))


def _safe_title(title: str) -> str:
    title = re.sub(r"[^\w\-. ]+", "", (title or "document")).strip() or "document"
    return title[:60]


# ─── Strip the chat wrapper so the file is JUST the document ─────────────────
# Answers arrive wrapped in chat affordances — a "download with the Export
# button" note, a trailing "Follow-ups" section, and conversational preamble /
# a title that names the file format. None of that belongs in the downloaded
# file. Every model wraps differently, so we normalize on export.

_FOLLOWUP_HEAD_RE = re.compile(
    r"^\s*#{0,6}\s*\*{0,2}\s*follow[\s\-]?ups?\s*\*{0,2}\s*:?\s*$", re.IGNORECASE)
_FORMAT_PAREN_RE = re.compile(
    r"\((?:markdown|md|word|docx|pdf|excel|xlsx|csv|text|txt|powerpoint|pptx)\)\s*$",
    re.IGNORECASE)
_FRAMING_RE = re.compile(
    r"^\s*(here(?:'?s| is| you go)|sure[,!.: ]|certainly[,!.: ]|of course[,!.: ]|"
    r"absolutely[,!.: ]|below is\b|i(?:'?ve| have) (?:created|prepared|revised|"
    r"updated|drafted|put together))",
    re.IGNORECASE)
_SEP_ONLY_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$")  # --- *** ___


def _is_download_note(line: str) -> bool:
    low = line.lower()
    if ("beneath this message" in low or "beneath your message" in low
            or "below this message" in low):
        return True
    return ("button" in low) and ("download" in low or "export" in low)


def _clean_for_export(md: str) -> str:
    """Drop the assistant's chat wrapper (download-button note, Follow-ups
    section, conversational preamble / format-title) so only the document
    remains. Falls back to the original if cleaning would empty it."""
    lines = (md or "").split("\n")

    # Cut a trailing "Follow-ups" section (its heading → end of message).
    for i, ln in enumerate(lines):
        if _FOLLOWUP_HEAD_RE.match(ln):
            lines = lines[:i]
            break

    # Remove any "download via the Export button" affordance line(s).
    lines = [ln for ln in lines if not _is_download_note(ln)]

    # Trim leading blanks / separators / conversational framing / format titles.
    while lines:
        s = lines[0].strip()
        bare = re.sub(r"^#{1,6}\s*", "", s).replace("*", "").strip()
        if (not s or _SEP_ONLY_RE.match(s) or _FRAMING_RE.match(s)
                or _FORMAT_PAREN_RE.search(bare)):
            lines.pop(0)
        else:
            break

    # Trim trailing blanks / separators.
    while lines and (not lines[-1].strip() or _SEP_ONLY_RE.match(lines[-1].strip())):
        lines.pop()

    cleaned = "\n".join(lines).strip()
    return cleaned or (md or "").strip()


def _clean_title(title: str) -> str:
    """Drop a trailing format tag like '(Markdown)' from a title/filename."""
    return _FORMAT_PAREN_RE.sub("", title or "").strip() or (title or "")


# The PDF core fonts are latin-1 only, so Unicode punctuation would otherwise
# become "?" — transliterate the common ones (bullets, dashes, smart quotes,
# non-breaking / thin spaces, arrows) to clean ASCII first.
_TRANSLIT = {
    0x2022: "-", 0x2023: "-", 0x25AA: "-", 0x25CF: "-", 0x2043: "-", 0x00B7: "-",
    0x2014: "-", 0x2013: "-", 0x2011: "-", 0x2212: "-",       # dashes / minus
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x2032: "'",       # single quotes
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x2033: '"',       # double quotes
    0x2026: "...", 0x2192: "->", 0x2190: "<-", 0x21D2: "=>", 0x21D0: "<=",
    0x00A0: " ", 0x2009: " ", 0x202F: " ", 0x2007: " ", 0x200B: "",  # spaces
    0x2013: "-", 0x2015: "-", 0xFE0F: "",
    # math / comparison operators (would otherwise show as "?")
    0x2264: "<=", 0x2265: ">=", 0x2260: "!=", 0x2248: "~=", 0x2261: "==",
    0x00D7: "x", 0x00F7: "/", 0x00B1: "+/-", 0x221E: "inf", 0x221A: "sqrt",
    0x00B0: " deg", 0x03BC: "u", 0x2211: "sum", 0x220F: "prod", 0x2208: "in",
    0x2200: "for all", 0x2203: "exists", 0x2205: "empty", 0x2229: "and",
    0x222A: "or", 0x00BD: "1/2", 0x00BC: "1/4", 0x00BE: "3/4", 0x2153: "1/3",
    0x2154: "2/3", 0x2013: "-", 0x2032: "'", 0x2033: '"',
}


def _latin(s: str) -> str:
    return (s or "").translate(_TRANSLIT).encode("latin-1", "replace").decode("latin-1")


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
                for ln in b["text"].split("\n"):
                    if ln.strip():
                        w.writerow([_strip_md(ln)])
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
    blocks = _parse_blocks(content)
    if not _skip_title(blocks, title):
        d.add_heading(title, 0)
    for b in blocks:
        t = b["type"]
        if t == "heading":
            d.add_heading(_strip_md(b["text"]), min(b["level"], 4))
        elif t == "para":
            # Preserve the answer's line breaks as line breaks in one paragraph
            # (tight spacing) instead of merging them into a run-on block; render
            # inline **bold** as real bold.
            plines = [ln for ln in b["text"].split("\n") if ln.strip()]
            if plines:
                p = d.add_paragraph()
                for idx, ln in enumerate(plines):
                    if idx:
                        p.add_run().add_break()
                    _docx_add_runs(p, ln)
        elif t == "list":
            for x in b["items"]:
                _docx_add_runs(d.add_paragraph(style="List Bullet"), x)
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


def _pdf_write(pdf, h: float, text: str, markdown: bool = False) -> None:
    """Write a block to the PDF, one source line per line (structure preserved),
    hard-wrapping very long unbreakable tokens so fpdf can always place them.
    With markdown=True, inline **bold** / __italic__ render as emphasis.

    new_x/new_y are explicit: each line must return to the left margin and move
    down, otherwise a second w=0 multi_cell starts at the right edge and fpdf
    raises "not enough horizontal space" (which would silently drop the line)."""
    from fpdf.enums import XPos, YPos
    for raw in (text or "").split("\n"):
        line = _latin(raw)
        # Insert a break opportunity inside tokens longer than ~90 chars.
        line = re.sub(r"(\S{90})", r"\1 ", line)
        try:
            pdf.multi_cell(0, h, line if line else " ", markdown=markdown,
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        except Exception:
            # Retry without markdown if a stray '*'/'_' trips the parser.
            try:
                pdf.multi_cell(0, h, line if line else " ",
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            except Exception:
                continue


def _pdf_table(pdf, rows: list) -> None:
    """Render a real bordered table (grid) with a bold header row. Falls back to
    pipe-joined lines if fpdf can't fit the layout (e.g. a very wide table)."""
    ncols = max((len(r) for r in rows), default=0)
    if ncols == 0:
        return
    data = [[_latin(_strip_md(c)) for c in r] + [""] * (ncols - len(r))
            for r in rows]
    try:
        with pdf.table(first_row_as_headings=True, text_align="LEFT",
                       line_height=6) as table:
            for r in data:
                trow = table.row()
                for cell in r:
                    trow.cell(cell)
    except Exception:
        pdf.set_font("Helvetica", "", 9)
        for r in rows:
            _pdf_write(pdf, 5, " | ".join(_strip_md(c) for c in r))


def _to_pdf(content: str, title: str):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    blocks = _parse_blocks(content)
    if not _skip_title(blocks, title):
        pdf.set_font("Helvetica", "B", 16)
        _pdf_write(pdf, 9, title)
        pdf.ln(2)
    for b in blocks:
        t = b["type"]
        if t == "heading":
            pdf.set_font("Helvetica", "B", max(11, 17 - b["level"]))
            _pdf_write(pdf, 7, _strip_md(b["text"]))
            pdf.ln(1)
        elif t == "para":
            pdf.set_font("Helvetica", "", 11)
            # Raw text + markdown so inline **bold** renders; line breaks kept.
            _pdf_write(pdf, 6, b["text"], markdown=True)
            pdf.ln(1)
        elif t == "list":
            pdf.set_font("Helvetica", "", 11)
            for x in b["items"]:
                _pdf_write(pdf, 6, "-  " + x, markdown=True)
            pdf.ln(1)
        elif t == "code":
            pdf.set_font("Courier", "", 8)
            _pdf_write(pdf, 4.5, b["text"])
            pdf.ln(1)
        elif t == "table" and b["rows"]:
            pdf.set_font("Helvetica", "", 9)
            _pdf_table(pdf, b["rows"])
            pdf.ln(2)
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
            if b["type"] == "heading":
                ws.append([_strip_md(b["text"])])
            elif b["type"] == "para":
                for ln in b["text"].split("\n"):
                    if ln.strip():
                        ws.append([_strip_md(ln)])
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
            state["bullets"].extend(
                _strip_md(ln) for ln in b["text"].split("\n") if ln.strip())
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


def export_document(content: str, fmt: str, title: str = "document",
                    clean: bool = True):
    """Return (bytes, mime_type, filename) for the requested format.

    clean=True (the in-response download) strips the chat wrapper so the file
    holds only the requested document. clean=False (the Export menu) keeps the
    whole AI response verbatim.
    """
    gen = _GENERATORS.get((fmt or "md").lower().strip())
    if gen is None:
        raise ValueError(f"Unsupported export format: {fmt}")
    body = _clean_for_export(content or "") if clean else (content or "")
    return gen(body, _safe_title(_clean_title(title)))
