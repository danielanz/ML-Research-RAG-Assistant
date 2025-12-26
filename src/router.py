from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Route:
    mode: str  # "qa" | "compare" | "method_card" | "claim_verify"

_COMPARE_PAT = re.compile(r"\b(compare|vs\.?|versus|difference|differences|similarities)\b", re.I)
_METHOD_PAT = re.compile(r"\b(method card|summarize method|architecture|pipeline|training objective)\b", re.I)
_VERIFY_PAT = re.compile(r"\b(verify|is it true|does the paper claim|evidence for|support the claim|refute)\b", re.I)

def route_query(question: str) -> Route:
    q = question.strip()
    if _VERIFY_PAT.search(q):
        return Route(mode="claim_verify")
    if _COMPARE_PAT.search(q):
        return Route(mode="compare")
    if _METHOD_PAT.search(q):
        return Route(mode="method_card")
    return Route(mode="qa")
