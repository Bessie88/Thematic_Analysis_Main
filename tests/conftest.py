"""Pytest hooks and shared fixtures for this package.

On restricted clusters, ``pip install -r requirements-dev.txt`` may fail because
numpy (and related wheels) are not published for that environment. These tests
only need pytest: ``pip install -r tests/requirements.txt``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def load_fixture(fixtures_dir: Path):
    def _load(name: str) -> Any:
        path = fixtures_dir / name
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return _load


@pytest.fixture
def gt_skills_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Writable skills directory with GT_USE_SKILLS enabled."""
    monkeypatch.setenv("GT_SKILLS_DIR", str(tmp_path))
    monkeypatch.setenv("GT_USE_SKILLS", "1")
    load_skill_text = __import__("agents.core.skills", fromlist=["load_skill_text"]).load_skill_text
    load_skill_text.cache_clear()
    return tmp_path
