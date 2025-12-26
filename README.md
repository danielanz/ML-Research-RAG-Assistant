# ML-Research-RAG-Assistant
Tool that ingests ML papers and answers research-grade questions including their citations.
Useful to easily dissect ML papers and query any questions you have to receive concrete answers whilst reducing hallucinations.

Local RAG assistant for ML papers with:
- PDF ingestion (page + section metadata)
- Chunk-level citations (page numbers + exact chunk text)
- Abstain behavior if unsupported
- Evaluation (Recall@K, MRR, citation coverage)
- Logging (JSONL)
- Streamlit UI