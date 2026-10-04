"""
Shared helpers for the example scripts: Art. 50 notice and run traces.

A trace is one JSON line per run in traces/<name>.jsonl (input, prompt, retrieval
hits, tool calls, output, tokens, latency) - searchable with grep/jq.
"""

import json
import os
import time
from datetime import datetime, timezone

TRACE_DIR = os.path.join(os.path.dirname(__file__), "traces")

AI_NOTICE = "Hinweis / Notice: You are interacting with an AI system (EU AI Act, Art. 50). Answers may be wrong."


def write_trace(name: str, record: dict) -> None:
    """Append one trace record to traces/<name>.jsonl."""
    os.makedirs(TRACE_DIR, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with open(os.path.join(TRACE_DIR, f"{name}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def usage_dict(response) -> dict:
    """Token usage of an OpenAI-style response (empty if the provider omits it)."""
    u = getattr(response, "usage", None)
    return {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens} if u else {}


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def ms(self) -> int:
        return int((time.perf_counter() - self.start) * 1000)
