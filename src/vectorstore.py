from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.chunking import Chunk

def build_embeddings(model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)

def chunks_to_documents(chunks: Iterable[Chunk]) -> list[Document]:
    docs: list[Document] = []
    for ch in chunks:
        md = dict(ch.metadata)
        md["chunk_id"] = ch.chunk_id
        docs.append(Document(page_content=ch.text, metadata=md))
    return docs

def get_chroma(
    persist_dir: Path,
    collection_name: str,
    embeddings: OpenAIEmbeddings,
) -> Chroma:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
