#!/usr/bin/env python3
"""One-shot fetch of approved codebook review from Supabase."""

from __future__ import annotations

import os
import sys

from agents.core.codebook_review import fetch_and_materialize_approved
from agents.core.paths import display_path, CODEBOOK_PATH


def main() -> int:
    slug = os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
    review_id = os.environ.get("CODEBOOK_REVIEW_ID", "").strip() or None
    try:
        path = fetch_and_materialize_approved(review_id=review_id, slug=slug)
    except RuntimeError as e:
        print(f"fetch_approved_codebook: {e}", file=sys.stderr)
        return 1

    print(f"fetch_approved_codebook: wrote {display_path(CODEBOOK_PATH)} from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
