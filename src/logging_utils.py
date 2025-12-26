from __future__ import annotations

from pathlib import Path
import json
import time
import uuid

def log_event(logs_dir: Path, event: dict) -> str:
    logs_dir.mkdir(parents=True, exist_ok=True)
    event_id = str(uuid.uuid4())
    payload = dict(event)
    payload["event_id"] = event_id
    payload["ts"] = time.time()

    log_path = logs_dir / "events.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return event_id
