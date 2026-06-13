#!/usr/bin/env python3
"""Poll Supabase until codebook review is approved; materialize local artifacts."""

from __future__ import annotations

import os
import sys

from agents.core.codebook_review import wait_for_approval


def main() -> int:
    slug = os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
    review_id = os.environ.get("CODEBOOK_REVIEW_ID", "").strip() or None
    try:
        row = wait_for_approval(slug, review_id=review_id)
    except TimeoutError as e:
        print(f"wait_for_codebook_review: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"wait_for_codebook_review: {e}", file=sys.stderr)
        return 1

    print(f"wait_for_codebook_review: approved review_id={row.get('id')} slug={slug!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
