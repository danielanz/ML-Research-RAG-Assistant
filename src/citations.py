from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

CIT_RE = re.compile(r"\[(?P<chunk_id>[0-9a-f]{12})\s+p\.(?P<page>\d+)\]")

@dataclass(frozen=True)
class CitedChunk:
    chunk_id: str
    page_number: int

def extract_citations(text: str) -> list[CitedChunk]:
    out: list[CitedChunk] = []
    for m in CIT_RE.finditer(text):
        out.append(CitedChunk(chunk_id=m.group("chunk_id"), page_number=int(m.group("page"))))
    return out

def format_citation(chunk_id: str, page_number: int) -> str:
    return f"[{chunk_id} p.{page_number}]"

def citation_ids(citations: Iterable[CitedChunk]) -> set[str]:
    return {c.chunk_id for c in citations}
