"""Tests for Supabase codebook review HTTP helpers (mocked)."""

import json
from unittest.mock import patch

from agents.core.supabase_http import (
    codebook_reviews_fetch_latest_approved,
    codebook_reviews_fetch_pending,
    codebook_reviews_insert_row,
)


@patch("agents.core.supabase_http.postgrest_request")
def test_insert_row(mock_post):
    mock_post.return_value = (201, json.dumps([{"id": "abc-123"}]))
    status, body = codebook_reviews_insert_row("https://x.supabase.co", "key", {"slug": "test"})
    assert status == 201
    assert "abc-123" in body


@patch("agents.core.supabase_http.postgrest_request")
def test_fetch_pending(mock_post):
    mock_post.return_value = (
        200,
        json.dumps([{"id": "r1", "status": "pending_review", "slug": "demo"}]),
    )
    row = codebook_reviews_fetch_pending("https://x.supabase.co", "key", "demo")
    assert row is not None
    assert row["id"] == "r1"


@patch("agents.core.supabase_http.postgrest_request")
def test_fetch_latest_approved(mock_post):
    mock_post.return_value = (
        200,
        json.dumps([{"id": "r2", "status": "approved", "codebook_v2": {"version": 1}}]),
    )
    row = codebook_reviews_fetch_latest_approved("https://x.supabase.co", "key", "demo")
    assert row["status"] == "approved"
    assert row["codebook_v2"]["version"] == 1
