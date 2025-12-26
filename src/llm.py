from __future__ import annotations

from langchain_openai import ChatOpenAI

def build_llm(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=float(temperature))
