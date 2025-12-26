from __future__ import annotations

from pathlib import Path
import json
import time

from src.config import load_config
from src.ingest_pdf import extract_pdf_pages
from src.chunking import chunk_pages
from src.vectorstore import build_embeddings, chunks_to_documents, get_chroma

def main() -> None:
    cfg = load_config()
    papers_dir = Path(cfg.papers_dir)
    chroma_dir = Path(cfg.chroma_dir)

    emb_model = cfg.raw["models"]["embeddings"]["model"]
    embeddings = build_embeddings(emb_model)

    vs = get_chroma(chroma_dir, collection_name="papers", embeddings=embeddings)

    pdfs = sorted([p for p in papers_dir.glob("*.pdf")])
    if not pdfs:
        raise SystemExit(f"No PDFs found in {papers_dir.resolve()}")

    total_added = 0
    for pdf in pdfs:
        pages = extract_pdf_pages(pdf, cfg.raw["ingestion"]["heading"])
        chunks = chunk_pages(pages, **cfg.raw["chunking"])
        docs = chunks_to_documents(chunks)

        # Avoid duplicate adds by chunk_id: simple deterministic guard
        # Chroma doesn't have a universal "upsert by metadata" in LangChain wrapper,
        # so we add all and rely on stable chunk_id + collection recreation for clean rebuild.
        ids = [d.metadata["chunk_id"] for d in docs]
        vs.add_documents(docs, ids=ids)
        total_added += len(docs)

        print(json.dumps({"pdf": pdf.name, "pages": len(pages), "chunks": len(docs)}, indent=2))

    # Note: langchain-chroma auto-persists when persist_directory is set
    print(f"Indexed {len(pdfs)} PDFs, added {total_added} chunks into {chroma_dir.resolve()}")

if __name__ == "__main__":
    main()
