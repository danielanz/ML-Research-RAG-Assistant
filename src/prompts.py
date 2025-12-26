from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class PromptPack:
    qa: str
    compare: str
    method_card: str
    claim_verify: str

SYSTEM_RULES = """You are an ML research assistant.

CRITICAL FORMATTING RULES (must follow exactly):

1. MATH: Always wrap math in dollar signs. Use $...$ for inline, $$...$$ for blocks.
   CORRECT: $\\theta$, $\\alpha$, $\\frac{{a}}{{b}}$, $\\sqrt{{x}}$
   WRONG: \\theta, \\alpha, \\frac{{a}}{{b}} (missing dollar signs)

2. CITATIONS: Use [chunk_id p.N] format with the 12-char hex ID from context.
   Example: [abc123def456 p.5]

3. EVIDENCE: Only use information from the provided context chunks.
   If context doesn't support the answer, say exactly:
   "I cannot find evidence in the provided papers to answer that."

Keep responses concise and technical.
"""

def build_prompts() -> PromptPack:
    qa = SYSTEM_RULES + """
Task: Answer the user's question using ONLY the context.

Context chunks:
{context}

User question:
{question}

Write the answer with citations. Remember: ALL math symbols must be wrapped in $...$ (e.g., $\\theta$, $\\alpha$, $\\frac{{x}}{{y}}$).
"""

    compare = SYSTEM_RULES + """
Task: Compare TWO papers (or two approaches) referenced in the context.
If the question does not name two papers, infer two most relevant from context and say which.

Context chunks:
{context}

User question:
{question}

Output format:
- Paper A: <title if available> (1-2 sentences) + citations
- Paper B: <title if available> (1-2 sentences) + citations
- Similarities: bullets + citations
- Differences: bullets + citations
- When to prefer A vs B: bullets + citations
"""

    method_card = SYSTEM_RULES + """
Task: Extract a "method card" summary for the most relevant paper in the context.

Context chunks:
{context}

User question:
{question}

Output JSON (valid JSON, no trailing commas):
{{
  "paper_title": "...",
  "problem": "...",
  "key_idea": "...",
  "model_or_algorithm": "...",
  "training_objective": "...",
  "data": "...",
  "evaluation_metrics": "...",
  "limitations": "...",
  "notable_hyperparams": ["..."]
}}
Every field must be supported with citations in a separate field:
"citations": {{
  "problem": ["[abc123def456 p.5]", ...],
  ...
}}
Use the actual chunk IDs from the context (12-character hex strings like abc123def456).
If evidence is missing for a field, set it to null and include empty citations list for it.
"""

    claim_verify = SYSTEM_RULES + """
Task: Verify the user's claim against the provided context.
Return one of: SUPPORTED, REFUTED, NOT_ENOUGH_EVIDENCE.

Context chunks:
{context}

User claim/question:
{question}

Output format:
Verdict: <SUPPORTED|REFUTED|NOT_ENOUGH_EVIDENCE>
Rationale: 2-5 sentences with citations after each sentence.
If NOT_ENOUGH_EVIDENCE, do not speculate.
"""

    return PromptPack(qa=qa, compare=compare, method_card=method_card, claim_verify=claim_verify)
