"""Call Supabase PostgREST with stdlib only (no supabase-py / no native deps)."""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _base_url(url: str) -> str:
    return url.strip().rstrip("/")


def _headers(api_key: str, *, with_content_type: bool) -> list[tuple[str, str]]:
    h = [
        ("apikey", api_key),
        ("Authorization", f"Bearer {api_key}"),
        ("Accept", "application/json"),
    ]
    if with_content_type:
        h.append(("Content-Type", "application/json"))
    return h


def postgrest_request(
    method: str,
    supabase_url: str,
    path: str,
    api_key: str,
    *,
    query: dict[str, str] | None = None,
    body: Any = None,
    timeout_sec: int = 300,
) -> tuple[int, str]:
    """
    path: e.g. 'pipeline_runs' (no leading slash).
    Returns (status_code, response_body_text).
    """
    base = _base_url(supabase_url)
    q = urllib.parse.urlencode(query) if query else ""
    url = f"{base}/rest/v1/{path}"
    if q:
        url = f"{url}?{q}"

    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, method=method.upper())
    for k, v in _headers(api_key, with_content_type=data is not None):
        req.add_header(k, v)

    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return e.code, raw


def _parse_json_array(body: str) -> list[dict]:
    if not body.strip():
        return []
    data = json.loads(body)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def pipeline_runs_select_smoke(supabase_url: str, api_key: str) -> tuple[int, str]:
    return postgrest_request(
        "GET",
        supabase_url,
        "pipeline_runs",
        api_key,
        query={"select": "id", "limit": "1"},
        timeout_sec=60,
    )


def pipeline_runs_insert_row(supabase_url: str, api_key: str, row: dict) -> tuple[int, str]:
    return postgrest_request(
        "POST",
        supabase_url,
        "pipeline_runs",
        api_key,
        body=[row],
        timeout_sec=300,
    )


def codebook_reviews_insert_row(supabase_url: str, api_key: str, row: dict) -> tuple[int, str]:
    return postgrest_request(
        "POST",
        supabase_url,
        "codebook_reviews",
        api_key,
        body=[row],
        query={"select": "id"},
        timeout_sec=300,
    )


def codebook_reviews_fetch_pending(
    supabase_url: str, api_key: str, slug: str
) -> dict | None:
    status, body = postgrest_request(
        "GET",
        supabase_url,
        "codebook_reviews",
        api_key,
        query={
            "select": "*",
            "slug": f"eq.{slug}",
            "status": "eq.pending_review",
            "order": "created_at.desc",
            "limit": "1",
        },
        timeout_sec=60,
    )
    if not (200 <= status < 300):
        raise RuntimeError(f"codebook_reviews fetch pending failed HTTP {status}: {body[:500]}")
    rows = _parse_json_array(body)
    return rows[0] if rows else None


def codebook_reviews_fetch_by_id(
    supabase_url: str, api_key: str, review_id: str
) -> dict | None:
    status, body = postgrest_request(
        "GET",
        supabase_url,
        "codebook_reviews",
        api_key,
        query={"select": "*", "id": f"eq.{review_id}", "limit": "1"},
        timeout_sec=60,
    )
    if not (200 <= status < 300):
        raise RuntimeError(f"codebook_reviews fetch by id failed HTTP {status}: {body[:500]}")
    rows = _parse_json_array(body)
    return rows[0] if rows else None


def codebook_reviews_fetch_latest_approved(
    supabase_url: str, api_key: str, slug: str
) -> dict | None:
    status, body = postgrest_request(
        "GET",
        supabase_url,
        "codebook_reviews",
        api_key,
        query={
            "select": "*",
            "slug": f"eq.{slug}",
            "status": "eq.approved",
            "order": "approved_at.desc",
            "limit": "1",
        },
        timeout_sec=60,
    )
    if not (200 <= status < 300):
        raise RuntimeError(f"codebook_reviews fetch approved failed HTTP {status}: {body[:500]}")
    rows = _parse_json_array(body)
    return rows[0] if rows else None


def codebook_reviews_poll_until_approved(
    supabase_url: str,
    api_key: str,
    slug: str,
    *,
    timeout_sec: int = 86400,
    interval_sec: int = 30,
    review_id: str | None = None,
) -> dict:
    """Poll until an approved review exists for slug (or specific review_id)."""
    start = time.monotonic()
    while True:
        if review_id:
            row = codebook_reviews_fetch_by_id(supabase_url, api_key, review_id)
            if row and row.get("status") == "approved" and row.get("codebook_v2"):
                return row
        else:
            row = codebook_reviews_fetch_latest_approved(supabase_url, api_key, slug)
            if row and row.get("codebook_v2"):
                return row

        if timeout_sec > 0 and (time.monotonic() - start) >= timeout_sec:
            raise TimeoutError(
                f"codebook review for slug={slug!r} not approved within {timeout_sec}s"
            )
        time.sleep(interval_sec)
