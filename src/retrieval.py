from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

@dataclass(frozen=True)
class Retrieved:
    doc: Document
    score: float  # similarity proxy in [0,1] (deterministic conversion)

def _distance_to_similarity(distance: float) -> float:
    # Chroma commonly returns distance where smaller is better.
    # Convert deterministically to a bounded similarity proxy.
    # This is not "true cosine similarity", but it's monotonic and stable.
    return 1.0 / (1.0 + max(0.0, float(distance)))

def retrieve(
    vs: Chroma,
    query: str,
    k: int,
    use_mmr: bool,
    mmr_fetch_k: int,
    mmr_lambda_mult: float,
) -> List[Retrieved]:
    if use_mmr:
        # MMR in LC wrappers returns docs without scores, so we do:
        # 1) get candidates with scores
        # 2) run MMR over texts using Chroma's retriever
        # Deterministic approach: use langchain's mmr retriever for doc selection,
        # then assign each selected doc its original similarity from candidate list.
        candidates = vs.similarity_search_with_score(query, k=mmr_fetch_k)
        cand_docs = [d for d, _ in candidates]
        mmr_docs = vs.max_marginal_relevance_search(
            query, k=k, fetch_k=mmr_fetch_k, lambda_mult=mmr_lambda_mult
        )
        # Map back to candidate scores by chunk_id
        cand_map = {d.metadata.get("chunk_id"): _distance_to_similarity(s) for d, s in candidates}
        out: List[Retrieved] = []
        for d in mmr_docs:
            sim = cand_map.get(d.metadata.get("chunk_id"), 0.0)
            out.append(Retrieved(doc=d, score=sim))
        return out

    pairs: List[Tuple[Document, float]] = vs.similarity_search_with_score(query, k=k)
    return [Retrieved(doc=d, score=_distance_to_similarity(s)) for d, s in pairs]

def should_abstain(retrieved: List[Retrieved], min_similarity: float) -> bool:
    if not retrieved:
        return True
    best = max(r.score for r in retrieved)
    return best < float(min_similarity)
