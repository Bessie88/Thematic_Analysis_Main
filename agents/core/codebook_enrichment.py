"""Enrich clustered codebook entries with definition, inclusion/exclusion criteria, and examples."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from .llm_client import make_chat_llm
from .skills import llm_invoke_with_skill
from .utils import clean_and_parse_json, log_step

_llm = make_chat_llm()

REQUIRED_KEYS = {"label", "definition", "keywords", "inclusion", "exclusion"}
BATCH_THRESHOLD = 25  # clusters above this size use hierarchical summarization
BATCH_SIZE = 20  # codes per batch in hierarchical mode


def _make_codebook_response_format(valid_ids: List[str]) -> dict:
    """Build a response format with enum-constrained code_ids for this cluster."""
    # Empty enum would force code_ids: [] always — fall back to plain string when no IDs given
    items_schema = (
        {"type": "string", "enum": sorted(valid_ids)} if valid_ids else {"type": "string"}
    )
    criterion_item = {
        "type": "object",
        "properties": {
            "criterion": {"type": "string"},
            "code_ids": {"type": "array", "items": items_schema},
        },
        "required": ["criterion", "code_ids"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "definition": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "inclusion": {"type": "array", "items": criterion_item},
            "exclusion": {"type": "array", "items": criterion_item},
        },
        "required": ["label", "definition", "keywords", "inclusion", "exclusion"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "codebook_entry", "schema": schema}}


# Kept for external imports that reference this name (e.g. enrich_dimensions.py legacy)
CODEBOOK_RESPONSE_FORMAT = _make_codebook_response_format([])


def _build_codes_text(
    codes: List[str],
    code_evidence: Dict[str, List[str]],
    code_notes: Dict[str, List[str]] = {},
    code_to_id: Dict[str, str] = {},
    max_quotes: int = 2,
) -> str:
    """Build codes text with local IDs (LC001, LC002...) to avoid cross-cluster confusion."""
    lines = []
    for local_idx, c in enumerate(codes):
        local_id = f"LC{local_idx + 1:03d}"
        lines.append(f"- [{local_id}] {c}")
        for ev in code_evidence.get(c, [])[:max_quotes]:
            lines.append(f'  [QUOTE] "{ev}"')
        for nt in code_notes.get(c, [])[:1]:
            lines.append(f"  [NOTE] {nt}")
    return "\n".join(lines)


def _enrich_batch(
    label: str,
    batch_codes: List[str],
    code_to_local_id: Dict[str, str],
    code_evidence: Dict[str, List[str]],
    code_notes: Dict[str, List[str]],
) -> dict:
    """Generate a partial codebook entry (inclusion/exclusion criteria) for one batch.
    Uses global LC IDs so code citations carry over directly to the synthesis step.
    """
    batch_ids = sorted(code_to_local_id[c] for c in batch_codes if c in code_to_local_id)
    valid_ids_str = ", ".join(batch_ids)

    lines = []
    for c in batch_codes:
        local_id = code_to_local_id.get(c, "")
        lines.append(f"- [{local_id}] {c}")
        for ev in code_evidence.get(c, [])[:2]:
            lines.append(f'  [QUOTE] "{ev}"')
        for nt in code_notes.get(c, [])[:1]:
            lines.append(f"  [NOTE] {nt}")
    codes_text = "\n".join(lines)

    items_schema = {"type": "string", "enum": batch_ids} if batch_ids else {"type": "string"}
    criterion_item = {
        "type": "object",
        "properties": {
            "criterion": {"type": "string"},
            "code_ids": {"type": "array", "items": items_schema},
        },
        "required": ["criterion", "code_ids"],
        "additionalProperties": False,
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "batch_codebook_entry",
            "schema": {
                "type": "object",
                "properties": {
                    "definition_notes": {"type": "string"},
                    "inclusion": {"type": "array", "items": criterion_item},
                    "exclusion": {"type": "array", "items": criterion_item},
                },
                "required": ["definition_notes", "inclusion", "exclusion"],
                "additionalProperties": False,
            },
        },
    }

    prompt = (
        f"Theme: {label}\n"
        f"VALID Code IDs for this batch — use ONLY these: {valid_ids_str}\n"
        f"Codes and participant quotes:\n{codes_text}\n\n"
        f"Based on these codes, identify:\n"
        f"1. definition_notes: key aspects of the theme these codes reveal\n"
        f"2. inclusion: criteria for what belongs under this theme (cite supporting code IDs)\n"
        f"3. exclusion: criteria for what does NOT belong here (cite supporting code IDs)\n"
        f"Focus on patterns within this batch only."
    )

    raw = llm_invoke_with_skill(
        _llm, "codebook_batch_enrich", prompt, response_format=response_format
    )
    try:
        result = clean_and_parse_json(raw)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {"definition_notes": "", "inclusion": [], "exclusion": []}


def _quote_in_evidence(quote: str, known: List[str]) -> bool:
    q = " ".join(quote.split()).lower()
    if not q:
        return True
    # Exact substring match always accepted
    for ev in known:
        ev_n = " ".join(ev.split()).lower()
        if q in ev_n or ev_n in q:
            return True
    # Short quotes (< 6 words) must be exact substring — fuzzy match too loose for code names
    q_words = set(q.split())
    if len(q_words) < 6:
        return False
    return (
        max(len(q_words & set(" ".join(ev.split()).lower().split())) / len(q_words) for ev in known)
        >= 0.80
    )


def _validate_entries(
    parsed: Dict[str, Any],
    valid_ids: set,
    id_to_code: Dict[str, str],
    code_evidence: Dict[str, List[str]],
    label: str,
) -> None:
    """Validate code_ids (list) and example quotes (list) against known evidence."""
    for section in ("inclusion", "exclusion"):
        for item in parsed.get(section, []):
            # 1. Filter code_ids to only valid ones
            raw_ids = item.get("code_ids", [])
            good_ids = [cid for cid in raw_ids if cid in valid_ids]
            if len(good_ids) < len(raw_ids):
                bad = set(raw_ids) - set(good_ids)
                log_step("ENRICH_ID_WARN", f"Cluster '{label}': invalid code_ids {bad} — removed")
            item["code_ids"] = good_ids

            # 2. Validate positional correspondence: examples[i] must come from code_ids[i]'s evidence
            examples = item.get("examples", [])
            paired = list(zip(good_ids, examples))  # align by position up to shorter length
            valid_pairs: List[tuple] = []
            for cid, ex in paired:
                known = code_evidence.get(id_to_code.get(cid, ""), [])
                if known and _quote_in_evidence(ex, known):
                    valid_pairs.append((cid, ex))
                else:
                    reason = "no evidence loaded" if not known else "not in evidence"
                    log_step(
                        "ENRICH_QUOTE_WARN",
                        f"Cluster '{label}': example for {cid} {reason} — dropped",
                    )
            # Any extra code_ids beyond paired examples are dropped (no matching example)
            if len(good_ids) > len(examples):
                log_step(
                    "ENRICH_ALIGN_WARN",
                    f"Cluster '{label}': {len(good_ids) - len(examples)} code_ids without examples — dropped",
                )
            item["code_ids"] = [cid for cid, _ in valid_pairs]
            item["examples"] = [ex for _, ex in valid_pairs]


def _enrich_one(
    cluster_id: str,
    label: str,
    codes: List[str],
    code_evidence: Dict[str, List[str]] = {},
    code_notes: Dict[str, List[str]] = {},
    code_to_id: Dict[str, str] = {},
) -> Dict[str, Any]:
    """Call LLM to enrich a single cluster; uses hierarchical summarization for large clusters."""
    try:
        local_id_to_code = {f"LC{i + 1:03d}": c for i, c in enumerate(codes)}
        valid_ids = set(local_id_to_code.keys())
        valid_ids_str = ", ".join(sorted(valid_ids))

        response_format = _make_codebook_response_format(list(valid_ids))

        if len(codes) <= BATCH_THRESHOLD:
            # ── Small cluster: single-shot ─────────────────────────────────────
            codes_with_evidence = [c for c in codes if code_evidence.get(c)]
            codes_without_evidence = [c for c in codes if not code_evidence.get(c)]
            log_step(
                "ENRICH_COVERAGE",
                f"Cluster {cluster_id} ({label}): {len(codes)} codes total — "
                f"all seen directly. "
                f"Evidence available: {len(codes_with_evidence)}/{len(codes)} codes. "
                + (f"No evidence for: {codes_without_evidence}" if codes_without_evidence else ""),
            )
            prompt = (
                f"Theme: {label}\n"
                f"VALID Code IDs — use ONLY these, do not invent others: {valid_ids_str}\n"
                f"Codes and participant quotes:\n"
                f"{_build_codes_text(codes, code_evidence, code_notes, code_to_id)}"
            )
            raw = llm_invoke_with_skill(
                _llm, "codebook_enrichment", prompt, response_format=response_format
            )
        else:
            # ── Large cluster: per-batch codebook enrich → synthesize ──────────
            batches = [codes[i : i + BATCH_SIZE] for i in range(0, len(codes), BATCH_SIZE)]
            code_to_local_id = {c: f"LC{i + 1:03d}" for i, c in enumerate(codes)}

            codes_with_evidence = [c for c in codes if code_evidence.get(c)]
            codes_without_evidence = [c for c in codes if not code_evidence.get(c)]

            # Each batch generates a structured partial codebook entry (with full evidence)
            with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as bex:
                partial_entries = list(
                    bex.map(
                        lambda b: _enrich_batch(
                            label, b, code_to_local_id, code_evidence, code_notes
                        ),
                        batches,
                    )
                )

            log_step(
                "ENRICH_COVERAGE",
                f"Cluster {cluster_id} ({label}): {len(codes)} codes total — batch path. "
                f"Evidence available: {len(codes_with_evidence)}/{len(codes)} codes. "
                f"All {len(codes)} codes seen with full evidence across {len(batches)} batches "
                f"(up to 2 quotes/code per batch). "
                + (
                    f"No evidence at all for: {codes_without_evidence}"
                    if codes_without_evidence
                    else ""
                ),
            )

            # Format partial entries for synthesis
            partial_block = ""
            for i, entry in enumerate(partial_entries):
                partial_block += f"\n[Batch {i + 1}/{len(batches)}]\n"
                partial_block += f"Definition notes: {entry.get('definition_notes', '')}\n"
                if entry.get("inclusion"):
                    partial_block += "Inclusion criteria:\n"
                    for c in entry["inclusion"]:
                        ids = ", ".join(c.get("code_ids", []))
                        partial_block += f"  - {c['criterion']} (codes: {ids})\n"
                if entry.get("exclusion"):
                    partial_block += "Exclusion criteria:\n"
                    for c in entry["exclusion"]:
                        ids = ", ".join(c.get("code_ids", []))
                        partial_block += f"  - {c['criterion']} (codes: {ids})\n"

            all_labels = "\n".join(f"- [{f'LC{i + 1:03d}'}] {c}" for i, c in enumerate(codes))

            prompt = (
                f"Theme: {label}\n"
                f"VALID Code IDs — use ONLY these, do not invent others: {valid_ids_str}\n\n"
                f"All {len(codes)} open codes in this cluster:\n{all_labels}\n\n"
                f"Partial codebook entries from each batch "
                f"(each generated from ~{BATCH_SIZE} codes with full evidence):\n"
                f"{partial_block}\n\n"
                f"Produce one final codebook entry by following these rules:\n"
                f"1. label: use the theme name as given above.\n"
                f"2. definition: synthesize a single coherent definition from the definition_notes "
                f"across all batches. It must be operational — a coder should be able to apply it "
                f"without ambiguity.\n"
                f"3. keywords: list key terms that characterise this theme.\n"
                f"4. inclusion: merge all inclusion criteria. If two criteria from different batches "
                f"say the same thing, keep one. If they are genuinely distinct, keep both. "
                f"Each criterion must cite at least one code_id from the valid list above.\n"
                f"5. exclusion: same merging rules as inclusion. Prioritise criteria that draw a "
                f"clear boundary against adjacent themes — vague criteria that merely restate the "
                f"definition should be dropped.\n"
                f"6. Never invent code IDs. If a criterion cannot be grounded in any valid code ID, "
                f"drop it."
            )
            raw = llm_invoke_with_skill(
                _llm, "codebook_enrichment", prompt, response_format=response_format
            )

        def _parse_and_validate(raw_text: str) -> tuple:
            p = clean_and_parse_json(raw_text)
            if not isinstance(p, dict) or not REQUIRED_KEYS.issubset(p):
                raise ValueError(f"missing keys: {REQUIRED_KEYS - set(p)}")
            p["label"] = label
            bad: set = set()
            for section in ("inclusion", "exclusion"):
                for item in p.get(section, []):
                    raw_ids = item.get("code_ids", [])
                    b = {cid for cid in raw_ids if cid not in valid_ids}
                    bad |= b
                    item["code_ids"] = [cid for cid in raw_ids if cid in valid_ids]
            # Drop criteria with no valid code_ids
            for section in ("inclusion", "exclusion"):
                p[section] = [item for item in p.get(section, []) if item.get("code_ids")]
            return p, bad

        parsed, bad_ids = _parse_and_validate(raw)

        # ── Retry once if invalid IDs were used ──────────────────────────────
        if bad_ids:
            log_step("ENRICH_RETRY", f"Cluster '{label}': invalid IDs {bad_ids} — retrying")
            retry_suffix = (
                f"\n\nCORRECTION: your previous response used non-existent IDs: "
                f"{', '.join(sorted(bad_ids))}. "
                f"Select ONLY from: {valid_ids_str}."
            )
            raw2 = llm_invoke_with_skill(
                _llm, "codebook_enrichment", prompt + retry_suffix, response_format=response_format
            )
            parsed, bad_ids2 = _parse_and_validate(raw2)
            if bad_ids2:
                log_step(
                    "ENRICH_WARN",
                    f"Cluster '{label}': still invalid IDs after retry {bad_ids2} — dropped",
                )

        # ── Fill examples programmatically from real evidence ─────────────────
        for section in ("inclusion", "exclusion"):
            for item in parsed.get(section, []):
                # Take evs[0] per cited code (open coding LLM already chose the best one),
                # deduplicate, cap at 2
                seen: set = set()
                examples: List[str] = []
                for cid in item.get("code_ids", []):
                    code = local_id_to_code.get(cid, "")
                    ev = next(iter(code_evidence.get(code, [])), None)
                    if ev and ev not in seen:
                        seen.add(ev)
                        examples.append(ev)
                    if len(examples) == 3:
                        break
                item["examples"] = examples
        # Drop criteria that ended up with no examples after programmatic fill
        for section in ("inclusion", "exclusion"):
            parsed[section] = [item for item in parsed.get(section, []) if item.get("examples")]
        return parsed

    except Exception as exc:
        log_step("ENRICH_WARN", f"Cluster {cluster_id} ({label}): {exc}")
        return {"label": label, "definition": "", "inclusion": "", "exclusion": "", "examples": []}


def enrich_codebook(
    cluster_names: Dict[str, str],
    cluster_to_codes: Dict[str, List[str]],
    workers: int = 4,
    code_evidence: Dict[str, List[str]] = {},
    code_notes: Dict[str, List[str]] = {},
    code_to_id: Dict[str, str] = {},
) -> Dict[str, Dict[str, Any]]:
    """Enrich all clusters in parallel; returns {cluster_id: full_entry}."""
    enriched: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _enrich_one,
                cid,
                cluster_names[cid],
                cluster_to_codes.get(cid, []),
                code_evidence,
                code_notes,
                code_to_id,
            ): cid
            for cid in cluster_names
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            enriched[cid] = fut.result()
            log_step("ENRICH_DONE", f"Cluster {cid}: {enriched[cid]['label']}")
    return enriched
