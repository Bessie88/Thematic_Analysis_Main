"""Challenge generation: for each (code, component), generate one challenge."""
from __future__ import annotations

import logging
import re
from typing import Optional

from .llm_client import call_llm, parse_json_response
from .prompts import (
    BENCHMARKER_SYSTEM,
    BENCHMARKER_USER_TEMPLATE,
    BENCHMARKER_INSTRUCTIONS_CRITERIA,
    BENCHMARKER_INSTRUCTIONS_CONCEPTUAL,
    COMPONENT_SCOPES,
)

CONCEPTUAL_COMPONENTS = {"label", "definition"}
from .schemas import ChallengeOutput, CodeEntry, ComponentType

logger = logging.getLogger(__name__)


def _format_criteria(criteria: list[dict]) -> str:
    if not criteria:
        return "(none)"
    lines = []
    for i, c in enumerate(criteria, 1):
        examples = "; ".join(c.get("examples", [])[:3])
        lines.append(f"  {i}. {c['criterion']}")
        if examples:
            lines.append(f"     Examples: {examples}")
    return "\n".join(lines)


def _format_criteria_with_ids(criteria: list[dict], prefix: str) -> tuple[str, dict[str, str]]:
    """
    Format criteria with explicit IDs (e.g. INC-1, EXC-2).
    Returns (formatted_text, id_to_text_map).
    """
    if not criteria:
        return "(none)", {}
    lines = []
    id_map: dict[str, str] = {}
    for i, c in enumerate(criteria, 1):
        cid = f"{prefix}-{i}"
        examples = "; ".join(c.get("examples", [])[:3])
        lines.append(f"  [{cid}] {c['criterion']}")
        if examples:
            lines.append(f"        Examples: {examples}")
        id_map[cid] = c["criterion"]
    return "\n".join(lines), id_map


def build_criterion_id_map(code: CodeEntry) -> dict[str, str]:
    """
    Return a complete map of all valid criterion IDs for a code entry.
    Keys: LABEL, DEF, INC-1..N, EXC-1..N
    Values: criterion text
    """
    id_map: dict[str, str] = {
        "LABEL": code.label,
        "DEF": code.definition,
    }
    for i, c in enumerate(code.inclusion, 1):
        id_map[f"INC-{i}"] = c["criterion"]
    for i, c in enumerate(code.exclusion, 1):
        id_map[f"EXC-{i}"] = c["criterion"]
    return id_map


def _build_numbered_evidence(
    code: CodeEntry,
    component: str = "",
    max_items: int = 20,
) -> tuple[str, dict[int, str]]:
    """
    Build a numbered evidence list and return (formatted_text, index_map).
    index_map: {1: "verbatim text", 2: "...", ...}
    For exclusion challenges, also appends neighbor cluster evidence.
    """
    items: list[str] = []
    for e in code.evidence_snippets[:max_items]:
        items.append(e)
    for c in code.inclusion + code.exclusion:
        for ex in c.get("examples", []):
            if ex not in items:
                items.append(ex)

    items = items[:max_items]

    # For exclusion: append evidence from neighboring clusters so the answerer/benchmarker
    # can reference where boundary cases actually belong
    if component == "exclusion" and code.neighbor_evidence_snippets:
        seen = set(items)
        for e in code.neighbor_evidence_snippets:
            if e not in seen and len(items) < max_items * 2:
                items.append(e)
                seen.add(e)

    index_map = {i + 1: text for i, text in enumerate(items)}
    lines = [f"[E{i}] {text}" for i, text in index_map.items()]
    return "\n".join(lines) or "(none)", index_map


def _resolve_indices(raw_indices: str, index_map: dict[int, str]) -> str:
    """
    Parse 'evidence_indices' from benchmarker response and resolve to real verbatim text.
    Returns a formatted string of the real evidence items.
    """
    nums = [int(n) for n in re.findall(r'\d+', raw_indices or "") if int(n) in index_map]
    if not nums:
        return ""
    parts = [f"[E{n}] {index_map[n]}" for n in nums]
    return "\n".join(parts)


def generate_challenge(
    code: CodeEntry,
    component: ComponentType,
    model: str,
    base_url: str = "http://localhost:8000/v1",
) -> Optional[ChallengeOutput]:
    evidence_text, index_map = _build_numbered_evidence(code, component=component)

    is_conceptual = component in CONCEPTUAL_COMPONENTS
    instructions_template = (
        BENCHMARKER_INSTRUCTIONS_CONCEPTUAL if is_conceptual
        else BENCHMARKER_INSTRUCTIONS_CRITERIA
    )
    challenge_instructions = instructions_template.format(component=component)

    user = BENCHMARKER_USER_TEMPLATE.format(
        label=code.label,
        definition=code.definition,
        inclusion_text=_format_criteria(code.inclusion),
        exclusion_text=_format_criteria(code.exclusion),
        evidence_text=evidence_text,
        component=component,
        scope=COMPONENT_SCOPES[component],
        challenge_instructions=challenge_instructions,
    )

    try:
        raw = call_llm(BENCHMARKER_SYSTEM, user, model=model, base_url=base_url)
        data = parse_json_response(raw)

        # Resolve cited indices → real verbatim text
        raw_indices = data.get("evidence_indices", "")
        evidence_used = _resolve_indices(str(raw_indices), index_map)
        if not evidence_used:
            evidence_used = data.get("evidence_used", "")

        return ChallengeOutput(
            challenge_question=data["challenge_question"],
            critique_claim=data["critique_claim"],
            evidence_used=evidence_used,
            failure_mode=data.get("failure_mode", "unsupported_or_vague"),
            failure_mechanism=data.get("failure_mechanism", ""),
        )
    except Exception as exc:
        logger.warning("Benchmarker failed for %s/%s: %s", code.code_id, component, exc)
        return None
