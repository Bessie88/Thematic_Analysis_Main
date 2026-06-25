"""Enrich codebook entries after clustering, before hierarchy construction.

Run AFTER open coding + clustering + refine and BEFORE --hierarchy-only:
    python -m agents.core.enrich_initial_codebook
    # or: python -m agents.cli --enrich-codebook-only --research-question "..."

Reads:  codebook.json (codebook + cluster_to_codes fields)
Writes: codebook.json (adds codebook_enriched; preserves short codebook labels)
"""

import sys

from .paths import CODEBOOK_PATH
from .qualitative_enrichment import run_cluster_qualitative_enrichment


def main() -> None:
    try:
        run_cluster_qualitative_enrichment()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Done. {CODEBOOK_PATH} updated with codebook_enriched.")


if __name__ == "__main__":
    main()
