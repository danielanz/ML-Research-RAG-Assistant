from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import hashlib
import tiktoken

from src.ingest_pdf import PageText, _heading_numbered

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict

def _stable_chunk_id(file_path: str, page_number: int, section_name: str, idx: int, text: str) -> str:
    h = hashlib.sha1()
    h.update(file_path.encode("utf-8"))
    h.update(str(page_number).encode("utf-8"))
    h.update(section_name.encode("utf-8"))
    h.update(str(idx).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:12]  # short, stable

def _pick_section_name(prev_section: str, headings: list[str]) -> str:
    # Deterministic: if multiple headings, take the last one on the page (often nearest top; but pages vary).
    # We choose FIRST heading if present to bias to top-of-page headings.
    if not headings:
        return prev_section
    return headings[0]

def _clean_heading(h: str) -> str:
    m = _heading_numbered.match(h.strip())
    if m:
        return m.group(3).strip()
    return h.strip()

def chunk_pages(
    pages: Iterable[PageText],
    chunk_tokens: int,
    chunk_overlap_tokens: int,
    min_chunk_tokens: int,
) -> list[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")

    chunks: list[Chunk] = []
    current_section = "Unknown"
    global_idx = 0

    for page in pages:
        current_section = _pick_section_name(current_section, page.detected_headings)
        section_name = _clean_heading(current_section)

        tokens = enc.encode(page.text)
        if not tokens:
            continue

        start = 0
        while start < len(tokens):
            end = min(start + chunk_tokens, len(tokens))
            window = tokens[start:end]
            if len(window) < min_chunk_tokens and end != len(tokens):
                # If too small and not last chunk, extend deterministically
                end = min(start + min_chunk_tokens, len(tokens))
                window = tokens[start:end]

            text = enc.decode(window).strip()
            if text:
                idx = global_idx
                global_idx += 1
                chunk_id = _stable_chunk_id(page.file_path, page.page_number, section_name, idx, text)

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=text,
                        metadata={
                            "source_file": page.file_path.split("/")[-1],
                            "source_path": page.file_path,
                            "page_number": page.page_number,
                            "section_name": section_name,
                            "chunk_index": idx,
                        },
                    )
                )

            if end == len(tokens):
                break
            start = max(0, end - chunk_overlap_tokens)

    return chunks
