#!/usr/bin/env python3
"""Enrich meta-theme dimensions with definition, inclusion/exclusion criteria, and examples.

Run AFTER --meta-themes-only has completed:
    python -m agents.core.enrich_dimensions
    # or: python -m agents.cli --enrich-dimensions-only --research-question "..."

Reads:  gt_meta_themes.json + codebook.json + gt_open_codes_all_reviews.md
Writes: gt_meta_themes_enriched.json
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List

from .llm_client import make_chat_llm
from .paths import META_THEMES_ENRICHED_PATH
from .skills import llm_invoke_with_skill
from .utils import clean_and_parse_json, log_step

REQUIRED_KEYS = {"label", "definition", "keywords", "inclusion", "exclusion"}
MAX_QUOTES_PER_CLUSTER = 3

_DIM_CRITERION_NO_EXAMPLE = {
    "type": "object",
    "properties": {
        "criterion": {"type": "string"},
        "code_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["criterion", "code_ids"],
    "additionalProperties": False,
}

_DIM_CRITERION_ITEM = {
    "type": "object",
    "properties": {
        "criterion": {"type": "string"},
        "code_ids": {"type": "array", "items": {"type": "string"}},
        "examples": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["criterion", "code_ids", "examples"],
    "additionalProperties": False,
}

_llm = make_chat_llm()


def _make_induction_format(valid_ids: List[str], all_cluster_ids: List[str] | None = None) -> dict:
    def _criterion_item(ids: List[str]) -> dict:
        items_schema = {"type": "string", "enum": sorted(ids)} if ids else {"type": "string"}
        return {
            "type": "object",
            "properties": {
                "criterion": {"type": "string"},
                "code_ids": {"type": "array", "items": items_schema},
            },
            "required": ["criterion", "code_ids"],
            "additionalProperties": False,
        }

    excl_ids = all_cluster_ids if all_cluster_ids else valid_ids
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "definition": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "inclusion": {"type": "array", "items": _criterion_item(valid_ids)},
            "exclusion": {"type": "array", "items": _criterion_item(excl_ids)},
        },
        "required": ["label", "definition", "keywords", "inclusion", "exclusion"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "dimension_entry_no_example", "schema": schema},
    }


def _quotes_block(
    cluster_ids: List[str],
    cluster_to_codes: Dict[str, List[str]],
    code_evidence: Dict[str, List[str]],
    cluster_labels: List[str] | None = None,
) -> str:
    lines = []
    for i, cid in enumerate(cluster_ids):
        padded = f"CL{cid.zfill(2)}"
        header = f"- [{padded}]" + (f" {cluster_labels[i]}" if cluster_labels else "")
        lines.append(header)
        codes = cluster_to_codes.get(cid, [])
        quotes_added = 0
        for code in codes:
            for quote in code_evidence.get(code, []):
                if quotes_added >= MAX_QUOTES_PER_CLUSTER:
                    break
                lines.append(f'  [QUOTE] "{quote}"')
                quotes_added += 1
            if quotes_added >= MAX_QUOTES_PER_CLUSTER:
                break
    return "\n".join(lines)


def _quote_in_cluster_evidence(
    quote: str,
    cluster_id: str,
    cluster_to_codes: Dict[str, List[str]],
    code_evidence: Dict[str, List[str]],
) -> bool:
    q = " ".join(quote.split()).lower()
    if not q:
        return True
    known: List[str] = []
    for code in cluster_to_codes.get(cluster_id, []):
        known.extend(code_evidence.get(code, []))
    for ev in known:
        ev_n = " ".join(ev.split()).lower()
        if q in ev_n or ev_n in q:
            return True
    if not known:
        return True
    q_words = set(q.split())
    if len(q_words) < 6:
        return False
    return (
        max(len(q_words & set(" ".join(ev.split()).lower().split())) / len(q_words) for ev in known)
        >= 0.80
    )


def _raw_cid(cid: str) -> str:
    return str(int(cid[2:])) if cid.startswith("CL") and cid[2:].isdigit() else cid


def _fill_examples(
    entry: Dict[str, Any],
    cluster_to_codes: Dict[str, List[str]],
    code_evidence: Dict[str, List[str]],
) -> None:
    used_globally: set = set()
    for section in ("inclusion", "exclusion"):
        for item in entry.get(section, []):
            examples: List[str] = []
            for cid in item.get("code_ids", []):
                raw_cid = _raw_cid(cid)
                for code in cluster_to_codes.get(raw_cid, []):
                    ev = next(iter(code_evidence.get(code, [])), None)
                    if ev and ev not in used_globally:
                        used_globally.add(ev)
                        examples.append(ev)
                        break
                if len(examples) == 3:
                    break
            item["examples"] = examples


def _enrich_one(
    name: str,
    cluster_ids: List[str],
    cluster_labels: List[str],
    cluster_to_codes: Dict[str, List[str]],
    code_evidence: Dict[str, List[str]],
) -> Dict[str, Any]:
    available_ids = ", ".join(f"CL{cid.zfill(2)}" for cid in cluster_ids)
    valid_ids = set(available_ids.replace(" ", "").split(","))
    all_cluster_ids = [
        f"CL{cid.zfill(2)}"
        for cid in sorted(cluster_to_codes.keys(), key=lambda x: int(x) if x.isdigit() else x)
    ]
    all_valid_ids = set(all_cluster_ids)
    induction_format = _make_induction_format(list(valid_ids), all_cluster_ids)

    def _run_pass1(prompt: str) -> Dict[str, Any]:
        raw = llm_invoke_with_skill(
            _llm, "dimension_criteria_induction", prompt, response_format=induction_format
        )
        parsed = clean_and_parse_json(raw)
        if not isinstance(parsed, dict) or not REQUIRED_KEYS.issubset(parsed):
            raise ValueError(f"Pass 1 missing keys: {REQUIRED_KEYS - set(parsed)}")
        parsed["label"] = name
        return parsed

    def _filter_and_collect_bad(parsed: Dict[str, Any]) -> set:
        bad_all: set = set()
        for section, allowed in (("inclusion", valid_ids), ("exclusion", all_valid_ids)):
            for item in parsed.get(section, []):
                raw_ids = item.get("code_ids", [])
                bad = {cid for cid in raw_ids if cid not in allowed}
                bad_all |= bad
                item["code_ids"] = [cid for cid in raw_ids if cid in allowed]
        return bad_all

    def _drop_empty_criteria(parsed: Dict[str, Any]) -> int:
        dropped = 0
        for section in ("inclusion", "exclusion"):
            before = parsed.get(section, [])
            after = [item for item in before if item.get("code_ids")]
            dropped += len(before) - len(after)
            parsed[section] = after
        return dropped

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            quotes_only_text = _quotes_block(cluster_ids, cluster_to_codes, code_evidence)
            all_ids_text = ", ".join(all_cluster_ids)
            base_prompt = (
                f"Dimension: {name}\n"
                f"This dimension's Cluster IDs (use for inclusion): {available_ids}\n"
                f"All available Cluster IDs (may use for exclusion boundary criteria): {all_ids_text}\n\n"
                f"Participant quotes by cluster (no labels — derive from the quotes):\n"
                f"{quotes_only_text}"
            )
            parsed = _run_pass1(base_prompt)
            bad_ids = _filter_and_collect_bad(parsed)
            dropped = _drop_empty_criteria(parsed)

            if bad_ids:
                log_step(
                    "DIM_ENRICH_RETRY",
                    f"'{name}': invalid IDs {bad_ids} — retrying with explicit feedback",
                )
                retry_prompt = (
                    f"{base_prompt}\n\n"
                    f"CORRECTION: your previous response used these non-existent IDs: "
                    f"{', '.join(sorted(bad_ids))}. "
                    f"For inclusion use only: {available_ids}. "
                    f"For exclusion use only: {all_ids_text}."
                )
                parsed = _run_pass1(retry_prompt)
                bad_ids2 = _filter_and_collect_bad(parsed)
                dropped += _drop_empty_criteria(parsed)
                if bad_ids2:
                    log_step(
                        "DIM_ENRICH_WARN",
                        f"'{name}': still invalid IDs after retry {bad_ids2} — dropped",
                    )

            if dropped:
                log_step(
                    "DIM_ENRICH_DROPPED",
                    f"'{name}': {dropped} criteria dropped (no valid code_ids)",
                )

            _fill_examples(parsed, cluster_to_codes, code_evidence)
            return parsed

        except Exception as exc:
            is_connection_err = "connection" in str(exc).lower() or "timeout" in str(exc).lower()
            if is_connection_err and attempt < max_attempts - 1:
                wait = 10 * (attempt + 1)
                log_step(
                    "DIM_ENRICH_RETRY",
                    f"'{name}': connection error on attempt {attempt + 1}/{max_attempts} "
                    f"— retrying in {wait}s: {exc}",
                )
                time.sleep(wait)
                continue
            log_step("DIM_ENRICH_WARN", f"'{name}': {exc}")
            return {
                "label": name,
                "definition": "",
                "keywords": [],
                "inclusion": [],
                "exclusion": [],
            }


def main() -> None:
    from .qualitative_enrichment import run_dimension_qualitative_enrichment

    try:
        run_dimension_qualitative_enrichment()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Done. Written to {META_THEMES_ENRICHED_PATH}")


if __name__ == "__main__":
    main()
