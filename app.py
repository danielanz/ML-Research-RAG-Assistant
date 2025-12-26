from __future__ import annotations

import json
import re
from pathlib import Path
import streamlit as st

from src.config import load_config
from src.pipeline import answer_question
from src.logging_utils import log_event


def format_answer_for_display(answer: str, cited_chunks: list[dict]) -> str:
    """Format the answer for display with nicer citations.
    
    - Converts raw citations [abc123def456 p.5] to numbered superscripts
    - LaTeX is expected to already be properly formatted by the LLM
    """
    # Build a map from chunk_id to a simple number
    chunk_to_num = {}
    for i, ch in enumerate(cited_chunks, 1):
        chunk_to_num[ch["chunk_id"]] = i
    
    # Replace citation format [chunk_id p.X] with [N] where N is the citation number
    def replace_citation(match):
        chunk_id = match.group(1)
        num = chunk_to_num.get(chunk_id, "?")
        return f"**[{num}]**"
    
    formatted = re.sub(r'\[([0-9a-f]{12})\s+p\.(\d+)\]', replace_citation, answer)
    
    return formatted

cfg = load_config()

st.set_page_config(page_title=cfg.raw["app"]["name"], layout="wide")
st.title(cfg.raw["app"]["name"])

tabs = st.tabs(["Chat", "Ingest", "Debug"])

with tabs[1]:
    st.header("Ingest papers (PDF)")
    st.write(f"Drop PDFs into `{cfg.papers_dir}` then run indexing.")
    st.code(f"uv run python scripts/index_papers.py")

    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        papers_dir = Path(cfg.papers_dir)
        papers_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            out = papers_dir / f.name
            out.write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploaded)} PDF(s) to {papers_dir.resolve()}. Now run indexing.")

with tabs[0]:
    st.header("Ask questions (with citations)")

    if "history" not in st.session_state:
        st.session_state.history = []

    q = st.text_input("Question", placeholder="E.g., Compare the training objectives of Paper A vs Paper B")
    if st.button("Answer") and q.strip():
        res = answer_question(q.strip())

        event_id = log_event(
            Path(cfg.logs_dir),
            {
                "type": "qa",
                "question": q.strip(),
                "mode": res.mode,
                "abstained": res.abstained,
                "answer": res.answer,
                "retrieved": res.retrieved,
                "cited_chunks": res.cited_chunks,
            },
        )

        st.session_state.history.append((q.strip(), res, event_id))

    for question, res, event_id in reversed(st.session_state.history):
        st.subheader(f"Q: {question}")
        st.caption(f"mode={res.mode} | abstained={res.abstained} | log_event={event_id}")
        
        # Format answer with nice citations and LaTeX rendering
        formatted_answer = format_answer_for_display(res.answer, res.cited_chunks)
        st.markdown(formatted_answer)

        if res.cited_chunks:
            with st.expander(f"📚 References ({len(res.cited_chunks)} citations)"):
                for i, ch in enumerate(res.cited_chunks, 1):
                    st.markdown(
                        f"**[{i}]** *{ch['source_file']}*, page {ch['page_number']} — Section: {ch['section_name']}"
                    )
                    st.code(ch["text"][:2000], language=None)

        with st.expander("Retrieved (debug)"):
            st.json(res.retrieved)

with tabs[2]:
    st.header("Logs")
    log_path = Path(cfg.logs_dir) / "events.jsonl"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()[-200:]
        st.code("\n".join(lines[-30:]))
        st.download_button("Download events.jsonl", data="\n".join(lines), file_name="events.jsonl")
    else:
        st.info("No logs yet. Ask a question first.")
