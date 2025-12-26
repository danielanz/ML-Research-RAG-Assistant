from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

@dataclass(frozen=True)
class AppConfig:
    raw: dict

    @property
    def seed(self) -> int:
        return int(self.raw["app"]["seed"])

    @property
    def papers_dir(self) -> Path:
        return Path(self.raw["paths"]["papers_dir"])

    @property
    def chroma_dir(self) -> Path:
        return Path(self.raw["paths"]["chroma_dir"])

    @property
    def logs_dir(self) -> Path:
        return Path(self.raw["paths"]["logs_dir"])

def load_config(path: str | Path = "config/app.yaml") -> AppConfig:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # Ensure required env vars exist early
    if raw["models"]["embeddings"]["provider"] == "openai" or raw["models"]["llm"]["provider"] == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY in environment (.env).")
    return AppConfig(raw=raw)
