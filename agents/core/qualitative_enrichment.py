"""Orchestrate qualitative codebook enrichment (cluster + meta-theme stages)."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from agents.core.evidence_io import (
    assign_open_code_ids,
    short_label,
    write_code_id_map,
)

from .codebook_enrichment import enrich_codebook
from .paths import (
    CODE_ID_MAP_PATH,
    CODEBOOK_PATH,
    DEFAULT_DATA_CSV,
    META_THEMES_ENRICHED_PATH,
    META_THEMES_PATH,
    OPEN_CODES_MARKDOWN_PATH,
    SOURCE_MEMORY_PATH,
    ensure_output_dirs,
)
from .source_memory import (
    ENRICHED_SCHEMA_VERSION,
    ground_enriched_entries,
    load_or_build_source_memory,
)
from .utils import log_step


def persist_cluster_enrichment(
    cb_data: dict,
    enriched: Dict[str, Dict[str, Any]],
    *,
    preserve_short_labels: bool = True,
) -> dict:
    """
    Merge qualitative entries into codebook.json payload.

    When preserve_short_labels is True, short labels in ``codebook`` are kept unchanged;
    full qualitative entries live under ``codebook_enriched``.
    """
    if preserve_short_labels:
        existing = cb_data.get("codebook", {})
        cb_data["codebook"] = {
            cid: short_label(str(existing.get(cid, entry.get("label", f"Cluster {cid}"))))
            for cid, entry in enriched.items()
        }
    else:
        cb_data["codebook"] = {
            cid: entry.get("label", f"Cluster {cid}") for cid, entry in enriched.items()
        }
    cb_data["codebook_enriched"] = enriched
    cb_data["enriched_schema_version"] = ENRICHED_SCHEMA_VERSION
    return cb_data


def _validate_cluster_enrichment(enriched: Dict[str, Dict[str, Any]]) -> List[str]:
    failed = []
    for cid, entry in enriched.items():
        if not entry.get("inclusion") and not entry.get("definition"):
            failed.append(f"{cid}:{entry.get('label', cid)}")
    return failed


def run_cluster_qualitative_enrichment(
    *,
    codebook_path: Path = CODEBOOK_PATH,
    md_path: Path = OPEN_CODES_MARKDOWN_PATH,
    csv_path: Path | None = None,
    workers: int | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Enrich cluster-level codebook entries; persist to codebook.json."""
    if not codebook_path.is_file():
        raise FileNotFoundError(f"Missing {codebook_path}; run refine step first.")

    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)

    cluster_to_codes: dict = cb_data.get("cluster_to_codes", {})
    codebook: dict = cb_data.get("codebook", {})
    if not cluster_to_codes:
        raise ValueError("no cluster_to_codes in codebook.json")

    cluster_names = {cid: short_label(str(label)) for cid, label in codebook.items()}
    data_csv = csv_path or Path(os.environ.get("GT_DATA_CSV", str(DEFAULT_DATA_CSV)))

    code_to_id = assign_open_code_ids(cluster_to_codes)
    write_code_id_map(code_to_id, CODE_ID_MAP_PATH)

    memory = load_or_build_source_memory(md_path, data_csv, SOURCE_MEMORY_PATH, CODE_ID_MAP_PATH)
    code_evidence, code_notes = memory.snippets_for_code_evidence()

    n_workers = workers if workers is not None else int(os.environ.get("GT_ENRICH_WORKERS", "4"))
    enriched = enrich_codebook(
        cluster_names,
        cluster_to_codes,
        workers=n_workers,
        code_evidence=code_evidence,
        code_notes=code_notes,
        code_to_id=code_to_id,
        source_memory=memory,
    )

    failed = _validate_cluster_enrichment(enriched)
    if failed:
        raise RuntimeError(
            f"{len(failed)} cluster(s) failed enrichment (empty result): {failed}. "
            "Check LLM server logs and gt_agent_trace.log; rerun --enrich-codebook-only."
        )

    cb_data = persist_cluster_enrichment(cb_data, enriched, preserve_short_labels=True)
    ensure_output_dirs()
    with open(codebook_path, "w", encoding="utf-8") as f:
        json.dump(cb_data, f, indent=2, ensure_ascii=False)

    log_step(
        "INITIAL_ENRICH_COMPLETE",
        f"{codebook_path.name} updated with {len(enriched)} codebook_enriched entries",
    )
    return enriched


def run_dimension_qualitative_enrichment(
    *,
    meta_themes_path: Path = META_THEMES_PATH,
    codebook_path: Path = CODEBOOK_PATH,
    out_path: Path = META_THEMES_ENRICHED_PATH,
    md_path: Path = OPEN_CODES_MARKDOWN_PATH,
    csv_path: Path | None = None,
    workers: int | None = None,
) -> List[Dict[str, Any]]:
    """Enrich meta-theme dimensions; write gt_meta_themes_enriched.json."""
    from .enrich_dimensions import _enrich_one

    if not meta_themes_path.is_file():
        raise FileNotFoundError(f"Missing {meta_themes_path}; run --meta-themes-only first.")
    if not codebook_path.is_file():
        raise FileNotFoundError(f"Missing {codebook_path}; run axial step first.")

    with open(meta_themes_path, encoding="utf-8") as f:
        meta_themes_data = json.load(f)
    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)

    codebook: Dict[str, str] = {
        str(cid): short_label(str(label)) for cid, label in cb_data.get("codebook", {}).items()
    }
    cluster_to_codes: Dict[str, List[str]] = cb_data.get("cluster_to_codes", {})
    meta_themes: List[Dict] = meta_themes_data.get("meta_themes", [])
    if not meta_themes:
        raise ValueError("no meta_themes found in gt_meta_themes.json")

    data_csv = csv_path or Path(os.environ.get("GT_DATA_CSV", str(DEFAULT_DATA_CSV)))
    memory = load_or_build_source_memory(md_path, data_csv, SOURCE_MEMORY_PATH, CODE_ID_MAP_PATH)
    code_evidence, _ = memory.snippets_for_code_evidence()

    n_workers = workers if workers is not None else int(os.environ.get("GT_ENRICH_WORKERS", "4"))
    enriched_list: List[Dict[str, Any] | None] = [None] * len(meta_themes)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for idx, mt in enumerate(meta_themes):
            name = mt.get("name", f"Dimension {idx}")
            cids = [str(cid) for cid in mt.get("cluster_ids", [])]
            cluster_labels = [codebook.get(cid, f"Cluster {cid}") for cid in cids]
            futures[
                ex.submit(
                    _enrich_one,
                    name,
                    cids,
                    cluster_labels,
                    cluster_to_codes,
                    code_evidence,
                    memory,
                )
            ] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            entry = fut.result()
            entry["cluster_ids"] = meta_themes[idx].get("cluster_ids", [])
            enriched_list[idx] = entry
            log_step("DIM_ENRICH_DONE", entry["label"])

    failed = [
        e["label"]
        for e in enriched_list
        if e and not e.get("inclusion") and not e.get("definition")
    ]
    if failed:
        raise RuntimeError(
            f"{len(failed)} meta-theme(s) failed enrichment (empty result): {failed}. "
            "Check LLM server logs and gt_agent_trace.log; rerun --enrich-dimensions-only."
        )

    out = {
        "meta_themes_enriched": enriched_list,
        "enriched_schema_version": ENRICHED_SCHEMA_VERSION,
    }
    ensure_output_dirs()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    log_step(
        "DIMENSION_ENRICH_COMPLETE", f"Wrote {out_path.name} ({len(enriched_list)} dimensions)"
    )
    return [e for e in enriched_list if e is not None]


def run_ground_enriched_only(
    *,
    codebook_path: Path = CODEBOOK_PATH,
    meta_path: Path = META_THEMES_ENRICHED_PATH,
    md_path: Path = OPEN_CODES_MARKDOWN_PATH,
    csv_path: Path | None = None,
) -> None:
    """Upgrade string examples to grounded objects without re-running enrichment LLMs."""
    from .enrich_dimensions import _raw_cid

    data_csv = csv_path or Path(os.environ.get("GT_DATA_CSV", str(DEFAULT_DATA_CSV)))
    memory = load_or_build_source_memory(md_path, data_csv, SOURCE_MEMORY_PATH, CODE_ID_MAP_PATH)

    if codebook_path.is_file():
        with open(codebook_path, encoding="utf-8") as f:
            cb_data = json.load(f)
        enriched = cb_data.get("codebook_enriched", {})
        cluster_to_codes = cb_data.get("cluster_to_codes", {})
        for cid, entry in enriched.items():
            if not isinstance(entry, dict):
                continue
            codes = cluster_to_codes.get(str(cid), [])
            local_id_to_code = {f"LC{i + 1:03d}": c for i, c in enumerate(codes)}
            ground_enriched_entries({str(cid): entry}, memory, local_id_to_code=local_id_to_code)
        cb_data["enriched_schema_version"] = ENRICHED_SCHEMA_VERSION
        with open(codebook_path, "w", encoding="utf-8") as f:
            json.dump(cb_data, f, indent=2, ensure_ascii=False)
        log_step("GROUND_ENRICHED", f"Grounded examples in {codebook_path.name}")

    if meta_path.is_file():
        cluster_to_codes: Dict[str, List[str]] = {}
        if codebook_path.is_file():
            with open(codebook_path, encoding="utf-8") as f:
                cluster_to_codes = json.load(f).get("cluster_to_codes", {})
        with open(meta_path, encoding="utf-8") as f:
            mt_data = json.load(f)
        entries = mt_data.get("meta_themes_enriched", [])
        ground_enriched_entries(
            entries,
            memory,
            cluster_to_codes=cluster_to_codes,
            raw_cid_fn=_raw_cid,
        )
        mt_data["enriched_schema_version"] = ENRICHED_SCHEMA_VERSION
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(mt_data, f, indent=2, ensure_ascii=False)
        log_step("GROUND_ENRICHED", f"Grounded examples in {meta_path.name}")
