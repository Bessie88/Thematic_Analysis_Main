#!/usr/bin/env python3
"""
Upload final pipeline artifacts to Supabase (PostgREST over HTTPS).
Reads:
  agents/outputs/data/codebook.json
  agents/outputs/data/gt_global_graph.json
  agents/outputs/data/research_report.md
  agents/outputs/data/gt_open_codes_all_reviews.md  (traceability: per-review codes + evidence)
  agents/outputs/data/gt_cooccurrence.json  (theme / meta-theme co-occurrence)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from agents.core.paths import (
    CODEBOOK_PATH,
    CODEBOOK_PROVENANCE_PATH,
    COOCCURRENCE_PATH,
    DATA_DIR,
    GLOBAL_GRAPH_PATH,
    OPEN_CODES_MARKDOWN_PATH,
    RESEARCH_REPORT_PATH,
)
from agents.core.supabase_http import pipeline_runs_insert_row


def _research_question_from_report_md(text: str) -> str:
    """Parse `## Research question` section from research_report.md."""
    marker = "## Research question"
    if marker not in text:
        return ""
    after = text.split(marker, 1)[1]
    for line in after.splitlines():
        s = line.strip()
        if not s:
            continue
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
            return s[1:-1].strip()
        return s
    return ""


def _research_question_from_graph(graph: dict) -> str:
    """Use root `tree.name` (same source as tree_assembly in the pipeline)."""
    tree = graph.get("tree")
    if not isinstance(tree, dict):
        return ""
    name = tree.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else ""


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def main() -> int:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        print(
            "upload_pipeline_to_supabase: missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY",
            file=sys.stderr,
        )
        return 1

    for label, path in (
        ("codebook", CODEBOOK_PATH),
        ("global graph", GLOBAL_GRAPH_PATH),
        ("research report", RESEARCH_REPORT_PATH),
        ("open codes markdown", OPEN_CODES_MARKDOWN_PATH),
        ("cooccurrence", COOCCURRENCE_PATH),
    ):
        if not path.is_file():
            print(f"upload_pipeline_to_supabase: missing {label}: {path}", file=sys.stderr)
            return 1

    slug = os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
    research_question = os.environ.get("RESEARCH_QUESTION", "").strip()

    codebook = _load_json(CODEBOOK_PATH)
    global_graph = _load_json(GLOBAL_GRAPH_PATH)
    report_md = _load_text(RESEARCH_REPORT_PATH)
    open_codes_md = _load_text(OPEN_CODES_MARKDOWN_PATH)
    cooccurrence = _load_json(COOCCURRENCE_PATH)

    if not research_question:
        research_question = _research_question_from_report_md(report_md)
    if not research_question:
        research_question = _research_question_from_graph(global_graph)

    meta: dict = {}
    if os.environ.get("SLURM_JOB_ID"):
        meta["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    if os.environ.get("GIT_COMMIT"):
        meta["git_commit"] = os.environ["GIT_COMMIT"]
    if CODEBOOK_PROVENANCE_PATH.is_file():
        try:
            with open(CODEBOOK_PROVENANCE_PATH, encoding="utf-8") as f:
                meta["codebook_provenance"] = json.load(f)
            meta["codebook_review_id"] = meta["codebook_provenance"].get("review_id")
        except (json.JSONDecodeError, OSError):
            pass

    row: dict = {
        "slug": slug,
        "research_question": research_question if research_question else None,
        "codebook": codebook,
        "global_graph": global_graph,
        "report_markdown": report_md,
        "open_codes_markdown": open_codes_md,
        "cooccurrence": cooccurrence,
    }
    if meta:
        row["meta"] = meta

    status, body = pipeline_runs_insert_row(url, key, row)
    if not (200 <= status < 300):
        print(f"upload_pipeline_to_supabase: insert failed HTTP {status}: {body[:2000]}", file=sys.stderr)
        bl = body.lower()
        if "row-level security" in bl or '"code":"42501"' in body or "42501" in body:
            print(
                "Hint: INSERT needs the Secret key (sb_secret_...) or legacy service_role JWT — "
                "not the Publishable/anon key. The smoke test SELECT can pass with anon; uploads cannot.",
                file=sys.stderr,
            )
        else:
            print(
                "Hint: create table pipeline_runs and policies (see agents/docs/SUPABASE_SETUP.md)",
                file=sys.stderr,
            )
        return 1

    print(f"upload_pipeline_to_supabase: inserted row for slug={slug!r} (data dir: {DATA_DIR})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
