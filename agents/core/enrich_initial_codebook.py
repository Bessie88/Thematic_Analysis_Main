"""Enrich codebook entries after clustering, before hierarchy construction.

Run AFTER open coding + LLM clustering + refine and BEFORE --hierarchy-only:
    python -m agents.scripts.enrich_initial_codebook

Reads:  codebook.json (codebook + cluster_to_codes fields)
Writes: codebook.json (adds codebook_enriched field in-place)
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.core.codebook_enrichment import enrich_codebook
from agents.core.paths import CODEBOOK_PATH, OPEN_CODES_MARKDOWN_PATH, DEFAULT_DATA_CSV, ensure_output_dirs
from agents.core.utils import log_step, parse_code_evidence


def main() -> None:
    if not CODEBOOK_PATH.is_file():
        print(f"Error: {CODEBOOK_PATH} not found. Run clustering step first.")
        sys.exit(1)

    with open(CODEBOOK_PATH, encoding="utf-8") as f:
        cb_data = json.load(f)

    cluster_to_codes: dict = cb_data.get("cluster_to_codes", {})
    codebook: dict = cb_data.get("codebook", {})
    if not cluster_to_codes:
        print("Error: no cluster_to_codes in codebook.json.")
        sys.exit(1)

    # Extract plain labels (strip rich-label suffixes if already present)
    cluster_names = {cid: label.split(" | ")[0] for cid, label in codebook.items()}

    csv_path = Path(os.environ.get("GT_DATA_CSV", str(DEFAULT_DATA_CSV)))
    code_evidence, code_notes = parse_code_evidence(OPEN_CODES_MARKDOWN_PATH, csv_path)
    print(f"Enriching {len(cluster_names)} codebook entries "
          f"(evidence for {len(code_evidence)} codes, notes for {len(code_notes)} codes loaded)...")

    # Assign stable IDs to all unique codes across all clusters
    seen: set = set()
    ordered_codes = []
    for cid in sorted(cluster_to_codes.keys(), key=lambda x: int(x) if x.isdigit() else x):
        for code in cluster_to_codes[cid]:
            if code not in seen:
                ordered_codes.append(code)
                seen.add(code)
    code_to_id = {code: f"OC{i + 1:03d}" for i, code in enumerate(ordered_codes)}

    id_map_path = CODEBOOK_PATH.parent / "gt_code_id_map.json"
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump({"code_to_id": code_to_id,
                   "id_to_code": {v: k for k, v in code_to_id.items()}}, f, indent=2, ensure_ascii=False)
    print(f"Assigned {len(code_to_id)} code IDs → {id_map_path}")

    workers = int(os.environ.get("GT_ENRICH_WORKERS", "4"))
    enriched = enrich_codebook(cluster_names, cluster_to_codes, workers=workers,
                               code_evidence=code_evidence, code_notes=code_notes,
                               code_to_id=code_to_id)

    def _rich_label(cid: str) -> str:
        e = enriched.get(cid, {})
        parts = [e.get("label", cluster_names[cid])]
        if e.get("definition"):
            parts.append(f"Definition: {e['definition']}")
        inc = e.get("inclusion", [])
        if isinstance(inc, list):
            inc = "; ".join(item.get("criterion", "") for item in inc if item.get("criterion"))
        if inc:
            parts.append(f"Inclusion: {inc}")
        exc = e.get("exclusion", [])
        if isinstance(exc, list):
            exc = "; ".join(item.get("criterion", "") for item in exc if item.get("criterion"))
        if exc:
            parts.append(f"Exclusion: {exc}")
        return " | ".join(parts)

    cb_data["codebook"] = {cid: _rich_label(cid) for cid in enriched}
    cb_data["codebook_enriched"] = enriched

    ensure_output_dirs()
    with open(CODEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(cb_data, f, indent=2, ensure_ascii=False)

    log_step("INITIAL_ENRICH_COMPLETE", f"codebook.json updated with {len(enriched)} enriched entries")
    print("Done. codebook.json updated with codebook and codebook_enriched.")


if __name__ == "__main__":
    main()
