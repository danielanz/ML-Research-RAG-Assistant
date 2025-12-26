from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from pathlib import Path
import json

from langchain_core.documents import Document

from src.config import load_config
from src.vectorstore import build_embeddings, get_chroma
from src.retrieval import retrieve, should_abstain, Retrieved
from src.prompts import build_prompts
from src.router import route_query
from src.llm import build_llm
from src.citations import extract_citations, citation_ids

@dataclass(frozen=True)
class AnswerResult:
    mode: str
    answer: str
    abstained: bool
    retrieved: list[dict]   # serializable
    cited_chunks: list[dict]  # serializable (includes exact text)

def _doc_to_payload(r: Retrieved) -> dict:
    md = r.doc.metadata
    return {
        "chunk_id": md.get("chunk_id"),
        "source_file": md.get("source_file"),
        "page_number": md.get("page_number"),
        "section_name": md.get("section_name"),
        "score": r.score,
        "text": r.doc.page_content,
    }

def _build_context(retrieved: list[Retrieved], max_chunks: int) -> str:
    parts = []
    for r in retrieved[:max_chunks]:
        md = r.doc.metadata
        parts.append(
            f"CHUNK {md.get('chunk_id')} | file={md.get('source_file')} | page={md.get('page_number')} | section={md.get('section_name')}\n"
            f"{r.doc.page_content}\n"
        )
    return "\n---\n".join(parts)

def answer_question(question: str) -> AnswerResult:
    cfg = load_config()
    prompts = build_prompts()

    embeddings = build_embeddings(cfg.raw["models"]["embeddings"]["model"])
    vs = get_chroma(Path(cfg.chroma_dir), "papers", embeddings)

    route = route_query(question)

    ret_cfg = cfg.raw["retrieval"]
    retrieved = retrieve(
        vs,
        query=question,
        k=int(ret_cfg["k"]),
        use_mmr=bool(ret_cfg["use_mmr"]),
        mmr_fetch_k=int(ret_cfg["mmr"]["fetch_k"]),
        mmr_lambda_mult=float(ret_cfg["mmr"]["lambda_mult"]),
    )

    abstain = should_abstain(retrieved, float(ret_cfg["min_similarity"]))
    if abstain:
        return AnswerResult(
            mode=route.mode,
            answer="I cannot find evidence in the provided papers to answer that.",
            abstained=True,
            retrieved=[_doc_to_payload(r) for r in retrieved],
            cited_chunks=[],
        )

    context = _build_context(retrieved, int(cfg.raw["prompts"]["max_context_chunks"]))

    if route.mode == "compare":
        template = prompts.compare
    elif route.mode == "method_card":
        template = prompts.method_card
    elif route.mode == "claim_verify":
        template = prompts.claim_verify
    else:
        template = prompts.qa

    llm = build_llm(cfg.raw["models"]["llm"]["model"], cfg.raw["models"]["llm"]["temperature"])
    prompt = template.format(context=context, question=question)
    response = llm.invoke(prompt).content

    # Extract citations & return the EXACT cited chunks used
    cits = extract_citations(response)
    cited_ids = citation_ids(cits)
    retrieved_map = {r.doc.metadata.get("chunk_id"): r.doc for r in retrieved}

    cited_payload = []
    for cid in cited_ids:
        d: Document | None = retrieved_map.get(cid)
        if d is None:
            # If model cites unknown chunk_id, we ignore it for "exact chunks" output
            continue
        cited_payload.append(
            {
                "chunk_id": cid,
                "source_file": d.metadata.get("source_file"),
                "page_number": d.metadata.get("page_number"),
                "section_name": d.metadata.get("section_name"),
                "text": d.page_content,
            }
        )

    # Enforce abstain if model failed to cite anything for non-trivial answers
    # Deterministic guard: if response is not the abstain string and has 0 valid citations => abstain.
    if response.strip() != "I cannot find evidence in the provided papers to answer that." and len(cited_payload) == 0:
        response = "I cannot find evidence in the provided papers to answer that."
        return AnswerResult(
            mode=route.mode,
            answer=response,
            abstained=True,
            retrieved=[_doc_to_payload(r) for r in retrieved],
            cited_chunks=[],
        )

    return AnswerResult(
        mode=route.mode,
        answer=response,
        abstained=False,
        retrieved=[_doc_to_payload(r) for r in retrieved],
        cited_chunks=cited_payload,
    )
