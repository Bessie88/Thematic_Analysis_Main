"""Tests for qualitative enrichment persistence (no LLM)."""

from agents.core.qualitative_enrichment import persist_cluster_enrichment


def test_persist_cluster_enrichment_preserves_short_labels():
    cb_data = {
        "codebook": {"0": "emotional strain", "1": "academic pressure"},
        "cluster_to_codes": {"0": ["code a"], "1": ["code b"]},
    }
    enriched = {
        "0": {
            "label": "emotional strain",
            "definition": "Feelings of depletion.",
            "keywords": ["anxiety"],
            "inclusion": [{"criterion": "x", "code_ids": ["LC001"], "examples": ["q"]}],
            "exclusion": [],
        },
        "1": {
            "label": "academic pressure",
            "definition": "Workload stress.",
            "keywords": ["overload"],
            "inclusion": [{"criterion": "y", "code_ids": ["LC001"], "examples": ["q2"]}],
            "exclusion": [],
        },
    }

    out = persist_cluster_enrichment(cb_data, enriched, preserve_short_labels=True)

    assert out["codebook"]["0"] == "emotional strain"
    assert out["codebook"]["1"] == "academic pressure"
    assert out["codebook_enriched"]["0"]["definition"] == "Feelings of depletion."
    assert "codebook_enriched" in out


def test_persist_cluster_enrichment_strips_existing_rich_labels():
    cb_data = {
        "codebook": {
            "0": "emotional strain | Definition: old def | Inclusion: old inc",
        },
        "cluster_to_codes": {"0": ["code a"]},
    }
    enriched = {
        "0": {
            "label": "emotional strain",
            "definition": "New definition.",
            "keywords": [],
            "inclusion": [],
            "exclusion": [],
        },
    }

    out = persist_cluster_enrichment(cb_data, enriched, preserve_short_labels=True)
    assert out["codebook"]["0"] == "emotional strain"
