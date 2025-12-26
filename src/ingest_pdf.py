from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import fitz  # PyMuPDF

@dataclass(frozen=True)
class PageText:
    file_path: str
    page_number: int  # 1-indexed
    text: str
    detected_headings: list[str]

_heading_numbered = re.compile(r"^\s*(\d+(\.\d+)*)\s+(.+)$")
_heading_allcaps = re.compile(r"^[A-Z0-9 \-,:]{6,}$")

def _is_heading_candidate(line: str, cfg: dict) -> bool:
    s = line.strip()
    if len(s) < cfg["min_len"] or len(s) > cfg["max_len"]:
        return False
    if cfg.get("require_no_period", True) and s.endswith("."):
        return False

    # Deterministic heuristics: numbered headings or ALLCAPS-like or Title Case short lines
    words = s.split()
    if len(words) > cfg["max_words"]:
        return False

    if _heading_numbered.match(s):
        return True
    if _heading_allcaps.match(s):
        return True

    # Title Case-ish: many words start with uppercase, few punctuation, short
    upper_starts = sum(1 for w in words if w[:1].isupper())
    ratio = upper_starts / max(1, len(words))
    punct = sum(1 for ch in s if ch in "[](){};")
    return ratio >= 0.6 and punct == 0

def extract_pdf_pages(pdf_path: Path, heading_cfg: dict) -> list[PageText]:
    doc = fitz.open(pdf_path)
    pages: list[PageText] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        # Normalize deterministically
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        detected: list[str] = []
        for line in text.split("\n"):
            if _is_heading_candidate(line, heading_cfg):
                detected.append(line.strip())

        pages.append(
            PageText(
                file_path=str(pdf_path),
                page_number=i + 1,
                text=text,
                detected_headings=detected,
            )
        )
    return pages
