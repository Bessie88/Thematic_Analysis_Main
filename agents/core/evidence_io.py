"""Load open-coding evidence from markdown artifacts and assign stable code IDs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .paths import CODE_ID_MAP_PATH, ensure_output_dirs


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
    code_evidence: Dict[str, List[str]] = defaultdict(list)
    code_notes: Dict[str, List[str]] = defaultdict(list)

    if not md_path.is_file():
        return dict(code_evidence), dict(code_notes)

    text = md_path.read_text(encoding="utf-8")
    blocks = re.split(r"^## Review \d+\s*$", text, flags=re.MULTILINE)
    for block in blocks:
        if not block.strip():
            continue
        for m in re.finditer(
            r"-\s*Code:\s*(.+?)\n\s+Evidence:\s*\"(.+?)\"(?:\n\s+Note:\s*(.+?))?(?=\n-|\n*$)",
            block,
            re.DOTALL | re.IGNORECASE,
        ):
            code = m.group(1).strip()
            evidence = m.group(2).strip()
            note = (m.group(3) or "").strip()
            if evidence and evidence not in code_evidence[code]:
                code_evidence[code].append(evidence)
            if note and note not in code_notes[code]:
                code_notes[code].append(note)

    # csv_path reserved for future CSV-backed evidence; not required today.
    _ = csv_path
    return dict(code_evidence), dict(code_notes)


def assign_open_code_ids(cluster_to_codes: Dict[str, List[str]]) -> Dict[str, str]:
    """Assign stable OC001-style IDs to unique open codes across all clusters."""
    all_codes: set[str] = set()
    for codes in cluster_to_codes.values():
        all_codes.update(codes)
    ordered_codes = sorted(all_codes)
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
