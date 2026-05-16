"""Pytest hooks for this package.

On restricted clusters, ``pip install -r requirements-dev.txt`` may fail because
numpy (and related wheels) are not published for that environment. These tests
only need pytest: ``pip install -r tests/requirements.txt``.
"""
