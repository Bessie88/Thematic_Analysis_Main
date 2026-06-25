"""Load open-coding evidence from markdown artifacts and assign stable code IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .paths import CODE_ID_MAP_PATH, ensure_output_dirs
from .source_memory import parse_open_coding_snippets


def short_label(label: str) -> str:
    """Strip rich-label suffixes (label | Definition: ...) for idempotent re-runs."""
    if not label:
        return label
    return label.split(" | ", 1)[0].strip()


def parse_code_evidence(
    md_path: Path,
    csv_path: Path | None = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Parse gt_open_codes_all_reviews.md into evidence and notes per open code.

    Returns:
        code_evidence: {code_label -> [quote, ...]}
        code_notes: {code_label -> [note, ...]}
    """
    _ = csv_path
    snippets = parse_open_coding_snippets(md_path)
    if not snippets:
        return {}, {}

    from collections import defaultdict

    code_evidence: Dict[str, List[str]] = defaultdict(list)
    code_notes: Dict[str, List[str]] = defaultdict(list)
    for s in snippets:
        if s.quote not in code_evidence[s.open_code]:
            code_evidence[s.open_code].append(s.quote)
        if s.note and s.note not in code_notes[s.open_code]:
            code_notes[s.open_code].append(s.note)
    return dict(code_evidence), dict(code_notes)


def assign_open_code_ids(cluster_to_codes: Dict[str, List[str]]) -> Dict[str, str]:
    """Assign stable OC001-style IDs to unique open codes across all clusters."""
    seen: set[str] = set()
    ordered_codes: List[str] = []
    for cid in sorted(cluster_to_codes.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        for code in cluster_to_codes.get(cid, []):
            if code not in seen:
                ordered_codes.append(code)
                seen.add(code)
    return {code: f"OC{i + 1:03d}" for i, code in enumerate(ordered_codes)}


def write_code_id_map(
    code_to_id: Dict[str, str],
    path: Path = CODE_ID_MAP_PATH,
) -> None:
    """Persist code_to_id and id_to_code mapping."""
    ensure_output_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"code_to_id": code_to_id, "id_to_code": {v: k for k, v in code_to_id.items()}},
            f,
            indent=2,
            ensure_ascii=False,
        )
