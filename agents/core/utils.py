"""Helpers for the GT pipeline: logging, text parsing, code extraction."""
import datetime
import json
import re
from typing import Any

from .paths import GT_AGENT_TRACE_LOG_PATH, ensure_output_dirs

LOG_FILE = GT_AGENT_TRACE_LOG_PATH


def log_step(step_name: str, content: Any):
    ensure_output_dirs()
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"\n[{timestamp}] === {step_name} ===\n{str(content)}\n" + "=" * 30
    print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def remove_think_tags(text: str) -> str:
    """Removes <think> blocks from reasoning models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def clean_and_parse_json(text: str):
    """Clean thoughts and extract JSON safely."""
    text_no_think = remove_think_tags(text)
    start_idx = text_no_think.find("{")
    end_idx = text_no_think.rfind("}")
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No JSON brackets found.")
    return json.loads(text_no_think[start_idx : end_idx + 1])


def extract_codes(open_coding_text: str) -> list[str]:
    """
    Extract code labels only from open coding output (no Evidence/Note).
    Expects lines like: "- Code: <code>"
    """
    if not open_coding_text or not open_coding_text.strip():
        return []
    codes = re.findall(r"^-\s*Code:\s*(.+)$", open_coding_text, re.MULTILINE | re.IGNORECASE)
    return [c.strip() for c in codes if c.strip()]
