"""Evaluation script for retrieval and grounding metrics.

Metrics computed:
- Recall@K: proportion of relevant chunks retrieved in top K
- MRR (Mean Reciprocal Rank): average of 1/rank for first relevant chunk
- Citation coverage: proportion of cited chunks that are actually relevant

Usage:
    uv run python scripts/evaluate.py

Requires:
- Indexed papers in data/chroma
- Labeled queries in evaluation/labeled_queries.jsonl with filled relevant_chunks
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

from src.config import load_config
from src.vectorstore import build_embeddings, get_chroma
from src.retrieval import retrieve
from src.router import route_query
from src.pipeline import answer_question


@dataclass
class RetrievalMetrics:
    recall_at_k: dict[int, float]  # {k: recall}
    mrr: float
    total_queries: int
    queries_with_relevant: int


@dataclass
class GroundingMetrics:
    citation_coverage: float  # proportion of answers that cite relevant chunks
    abstention_accuracy: float  # proportion of correct abstentions
    total_queries: int


def load_labeled_queries(path: Path) -> list[dict]:
    """Load labeled queries from JSONL file."""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@K: what fraction of relevant docs are in top K."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def compute_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute reciprocal rank: 1/rank of first relevant doc."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(
    queries: list[dict],
    vs,
    k_values: list[int],
    retrieval_cfg: dict,
) -> RetrievalMetrics:
    """Evaluate retrieval quality on labeled queries."""
    recalls = {k: [] for k in k_values}
    mrrs = []
    queries_with_relevant = 0

    max_k = max(k_values)

    for q in queries:
        relevant_ids = set(q.get("relevant_chunks", []))
        if not relevant_ids:
            continue  # Skip queries without labeled relevant chunks

        queries_with_relevant += 1

        # Retrieve documents
        retrieved = retrieve(
            vs,
            query=q["query"],
            k=max_k,
            use_mmr=bool(retrieval_cfg.get("use_mmr", False)),
            mmr_fetch_k=int(retrieval_cfg.get("mmr", {}).get("fetch_k", 24)),
            mmr_lambda_mult=float(retrieval_cfg.get("mmr", {}).get("lambda_mult", 0.65)),
        )

        retrieved_ids = [r.doc.metadata.get("chunk_id") for r in retrieved]

        # Compute metrics for each k
        for k in k_values:
            recalls[k].append(compute_recall_at_k(retrieved_ids, relevant_ids, k))

        mrrs.append(compute_reciprocal_rank(retrieved_ids, relevant_ids))

    # Average metrics
    avg_recalls = {k: sum(v) / len(v) if v else 0.0 for k, v in recalls.items()}
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

    return RetrievalMetrics(
        recall_at_k=avg_recalls,
        mrr=avg_mrr,
        total_queries=len(queries),
        queries_with_relevant=queries_with_relevant,
    )


def evaluate_grounding(queries: list[dict]) -> GroundingMetrics:
    """Evaluate citation/grounding quality.

    This runs the full pipeline and checks:
    - Whether cited chunks are among relevant chunks
    - Whether abstention happens appropriately
    """
    citation_hits = []
    abstention_correct = []

    for q in queries:
        relevant_ids = set(q.get("relevant_chunks", []))
        should_abstain = len(relevant_ids) == 0

        try:
            result = answer_question(q["query"])

            if result.abstained:
                # Check if abstention was correct
                abstention_correct.append(should_abstain)
            else:
                abstention_correct.append(not should_abstain)

                # Check citation coverage
                cited_ids = {c["chunk_id"] for c in result.cited_chunks}
                if cited_ids and relevant_ids:
                    coverage = len(cited_ids & relevant_ids) / len(cited_ids)
                    citation_hits.append(coverage)
                elif not relevant_ids:
                    # No relevant chunks labeled, can't evaluate
                    pass
                else:
                    citation_hits.append(0.0)
        except Exception as e:
            print(f"Error evaluating query '{q['query'][:50]}...': {e}")
            continue

    return GroundingMetrics(
        citation_coverage=sum(citation_hits) / len(citation_hits) if citation_hits else 0.0,
        abstention_accuracy=sum(abstention_correct) / len(abstention_correct) if abstention_correct else 0.0,
        total_queries=len(queries),
    )


def evaluate_router(queries: list[dict]) -> dict[str, float]:
    """Evaluate query router accuracy."""
    correct = 0
    total = 0

    for q in queries:
        expected_mode = q.get("expected_mode")
        if not expected_mode:
            continue

        route = route_query(q["query"])
        if route.mode == expected_mode:
            correct += 1
        total += 1

    return {
        "router_accuracy": correct / total if total > 0 else 0.0,
        "total_evaluated": total,
    }


def main() -> None:
    cfg = load_config()

    # Load labeled queries
    queries_path = Path("evaluation/labeled_queries.jsonl")
    if not queries_path.exists():
        raise SystemExit(f"Labeled queries not found at {queries_path}")

    queries = load_labeled_queries(queries_path)
    print(f"Loaded {len(queries)} labeled queries")

    # Setup vector store
    embeddings = build_embeddings(cfg.raw["models"]["embeddings"]["model"])
    vs = get_chroma(Path(cfg.chroma_dir), "papers", embeddings)

    # Evaluate router
    print("\n=== Router Evaluation ===")
    router_metrics = evaluate_router(queries)
    print(f"Router accuracy: {router_metrics['router_accuracy']:.2%}")
    print(f"Queries evaluated: {router_metrics['total_evaluated']}")

    # Evaluate retrieval
    print("\n=== Retrieval Evaluation ===")
    k_values = cfg.raw["evaluation"]["k_values"]
    retrieval_metrics = evaluate_retrieval(
        queries,
        vs,
        k_values=k_values,
        retrieval_cfg=cfg.raw["retrieval"],
    )

    print(f"Queries with labeled relevant chunks: {retrieval_metrics.queries_with_relevant}/{retrieval_metrics.total_queries}")

    if retrieval_metrics.queries_with_relevant > 0:
        print(f"MRR: {retrieval_metrics.mrr:.4f}")
        for k, recall in sorted(retrieval_metrics.recall_at_k.items()):
            print(f"Recall@{k}: {recall:.4f}")
    else:
        print("No queries have labeled relevant_chunks. Fill them in to evaluate retrieval.")

    # Evaluate grounding (runs full pipeline - may cost API calls)
    print("\n=== Grounding Evaluation ===")
    run_grounding = input("Run grounding evaluation? This calls the LLM. [y/N]: ").strip().lower()
    if run_grounding == "y":
        grounding_metrics = evaluate_grounding(queries)
        print(f"Citation coverage: {grounding_metrics.citation_coverage:.2%}")
        print(f"Abstention accuracy: {grounding_metrics.abstention_accuracy:.2%}")
    else:
        print("Skipped grounding evaluation.")

    # Save results
    results = {
        "router": router_metrics,
        "retrieval": {
            "mrr": retrieval_metrics.mrr,
            "recall_at_k": retrieval_metrics.recall_at_k,
            "queries_with_relevant": retrieval_metrics.queries_with_relevant,
        },
    }

    output_path = Path("evaluation/results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

