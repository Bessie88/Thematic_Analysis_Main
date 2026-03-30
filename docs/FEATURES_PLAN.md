# Post-Pipeline Analysis Features — Implementation Plan

Three new standalone analysis modules to add after the main thematic analysis pipeline completes. Each produces its own JSON output and is invocable via a new CLI flag.

---

## Overview

| Feature | CLI Flag | Output File | LLM Required |
|---|---|---|---|
| Evidence Tracing | `--evidence-only` | `gt_evidence_library.json` | No |
| Co-occurrence Network | `--cooccurrence-only` | `gt_cooccurrence.json` | No |
| Outlier Detection | `--outliers-only` | `gt_outliers.json` | No |

All three are **pure post-processing** — they consume existing pipeline outputs and produce new JSON files. No LLM calls needed.

---

## Feature 1: Evidence Tracing

### What
For every node in the thematic tree (meta-themes, themes, sub-themes, and codes), surface the best supporting quotes from the original reviews. Connects the abstract hierarchy back to the raw text, making the analysis auditable and publishable.

### Why
The open coding LLM already extracts evidence quotes per code (`Evidence: "..."` in `gt_open_codes_all_reviews.md`), but `extract_codes()` in `utils.py` discards them. The final tree has no connection back to the raw text.

### Algorithm
1. Parse `gt_open_codes_all_reviews.md` by splitting on `## Review N` headers; regex-extract `(code, evidence_quote)` pairs per review
2. Apply `dedup_map` from `gt_clustered_codes.json` to map original code strings → canonical codes
3. Walk `gt_global_graph.json` tree recursively to build: `canonical_code → {sub_theme, theme, meta_theme}`
4. Aggregate: for each tree node, collect all evidence quotes and review_ids from descendant codes (cap at 5 quotes per node)

### Input Files Required
- `gt_open_codes_all_reviews.md` — raw LLM open coding output with evidence quotes
- `gt_clustered_codes.json` — `dedup_map` for original → canonical code mapping
- `gt_global_graph.json` — tree for code → node ancestry mapping

### Output: `gt_evidence_library.json`
```json
{
  "by_node": {
    "Meta-Theme A": {
      "type": "meta_theme",
      "quote_count": 142,
      "review_ids": [1, 4, 7],
      "top_quotes": ["climate change is existential", "we are out of time"]
    },
    "dismissal of scientific consensus": {
      "type": "code",
      "quote_count": 3,
      "review_ids": [12],
      "top_quotes": ["scientists are just guessing"]
    }
  }
}
```

### Implementation
- **New file**: `agents/evidence_tracing.py` (~180 lines, standalone script)
- **Modify**: `agents/core/paths.py` — add `EVIDENCE_LIBRARY_PATH`
- **Modify**: `agents/cli.py` — add `--evidence-only` flag

---

## Feature 2: Co-occurrence Network

### What
Using `codes_per_review`, count how often each pair of themes/meta-themes is mentioned in the same review. Reveals latent relationships between themes that the hierarchy alone cannot show.

### Why
The hierarchy is a tree (each code has one parent), so cross-cutting relationships are invisible. Co-occurrence adds a quantitative relational layer: e.g., "climate denial" and "economic anxiety" always co-occur.

### Algorithm
1. Walk `gt_global_graph.json` tree once to build two lookup dicts:
   - `code_to_theme: {code_str → theme_name}`
   - `code_to_meta_theme: {code_str → meta_theme_name}`
2. For each review in `gt_clustered_codes.json`'s `codes_per_review` (canonical codes):
   - Map each code → its theme and meta_theme
   - Get unique set of themes/meta_themes touched by this review
   - Increment co-occurrence count for every pair in that set
3. Compute review coverage per theme/meta-theme
4. Sort pairs by count descending

### Input Files Required
- `gt_clustered_codes.json` — `codes_per_review` with canonical codes (1-indexed review IDs)
- `gt_global_graph.json` — tree for code → ancestry mapping

### Output: `gt_cooccurrence.json`
```json
{
  "meta_theme_matrix": {
    "Denial and Dismissal": {"Urgency Framing": 234, "Scientific Discourse": 89}
  },
  "theme_matrix": {
    "climate change denial tactics": {"urgency framing": 45}
  },
  "top_meta_theme_pairs": [
    {"pair": ["Denial and Dismissal", "Urgency Framing"], "count": 234, "pct_reviews": 0.234}
  ],
  "top_theme_pairs": [],
  "review_coverage": {
    "meta_themes": {"Denial and Dismissal": {"count": 450, "pct": 0.45}},
    "themes": {}
  },
  "total_reviews": 1000
}
```

### Implementation
- **New file**: `agents/cooccurrence.py` (~130 lines, standalone script)
- **Modify**: `agents/core/paths.py` — add `COOCCURRENCE_PATH`
- **Modify**: `agents/cli.py` — add `--cooccurrence-only` flag

---

## Feature 3: Outlier Detection

### What
Re-embed all canonical codes using the same Qwen3-Embedding model, compute silhouette scores (distance to own cluster centroid vs. nearest other centroid), and flag codes where `silhouette < 0` — meaning the code is closer to another cluster than its own. Also flags reviews where > 50% of their codes are outliers.

### Why
K-means forces every code into a cluster even if it doesn't fit. Outlier detection reveals: misclassified codes (potential quality issue), genuinely novel concepts (potential new cluster), and unusual reviews (edge cases or noise).

### Algorithm
1. Load `all_codes` (ordered list) and `labels` (cluster assignment per code) from `gt_clustered_codes.json`
2. Re-embed all codes using same model path resolution as `tools.py`:
   - `GT_EMBED_MODEL` env var → local `agents/weights/Qwen3-Embedding-0.6B` → HuggingFace
3. Compute cluster centroids: `centroid[k] = mean(embeddings[labels == k])`
4. For each code:
   - `dist_own = 1 - cosine_sim(embedding[i], centroid[labels[i]])`
   - `dist_nearest = min(1 - cosine_sim(embedding[i], centroid[k]) for k != labels[i])`
   - `silhouette = (dist_nearest - dist_own) / max(dist_own, dist_nearest)`
5. Flag codes with `silhouette < 0` as outliers
6. Map outlier codes → `review_ids` via `codes_per_review`
7. Flag reviews where `outlier_codes / total_codes > 0.5`

### Input Files Required
- `gt_clustered_codes.json` — `all_codes`, `labels`, `codes_per_review`, cluster assignments
- Qwen3-Embedding-0.6B model (same as used in axial coding)

### Output: `gt_outliers.json`
```json
{
  "summary": {
    "total_codes": 2121,
    "outlier_codes": 78,
    "outlier_pct": 0.037,
    "total_reviews": 1000,
    "outlier_reviews": 23
  },
  "outlier_codes": [
    {
      "code": "some code that doesn't fit",
      "cluster_id": "3",
      "cluster_label": "Climate Urgency Framing",
      "silhouette_score": -0.18,
      "dist_to_centroid": 0.42,
      "nearest_other_cluster_id": "1",
      "nearest_other_cluster_label": "Denial Tactics",
      "dist_to_nearest_other": 0.31,
      "review_ids": [12, 45]
    }
  ],
  "outlier_reviews": [
    {
      "review_id": 45,
      "outlier_code_fraction": 0.67,
      "codes": ["code1", "code2", "code3"],
      "outlier_codes": ["code1", "code3"]
    }
  ]
}
```

### Implementation
- **New file**: `agents/outlier_detection.py` (~160 lines, standalone script)
- **Modify**: `agents/core/paths.py` — add `OUTLIERS_PATH`
- **Modify**: `agents/cli.py` — add `--outliers-only` flag

---

## Shared Changes Across All Features

### `agents/core/paths.py`
```python
EVIDENCE_LIBRARY_PATH = DATA_DIR / "gt_evidence_library.json"
COOCCURRENCE_PATH     = DATA_DIR / "gt_cooccurrence.json"
OUTLIERS_PATH         = DATA_DIR / "gt_outliers.json"
```

### `agents/cli.py`
```python
p.add_argument("--evidence-only",      action="store_true", help="Build evidence quote library from open coding markdown.")
p.add_argument("--cooccurrence-only",  action="store_true", help="Compute theme co-occurrence matrix from codes_per_review.")
p.add_argument("--outliers-only",      action="store_true", help="Detect outlier codes and reviews using silhouette scoring.")
```

### Helper functions to reuse (no reimplementation needed)
| Function | Location |
|---|---|
| `log_step()` | `agents/core/utils.py` |
| `normalize_label()` | `agents/codebook_cleanup.py` |
| `load_datapoint_frequency()` | `agents/codebook_cleanup.py` |
| `WEIGHTS_DIR`, `DATA_DIR`, `display_path()` | `agents/core/paths.py` |

---

## Verification Steps

| Feature | How to verify |
|---|---|
| Evidence Tracing | Every leaf code should have ≥ 1 quote; every meta-theme should have multiple propagated quotes; quotes should be recognizable substrings from original reviews |
| Co-occurrence | Matrix should be symmetric; top pairs should make intuitive thematic sense; pct_reviews values should be < 1.0 and sum > 1.0 (reviews can touch multiple themes) |
| Outlier Detection | Outlier codes should visibly not belong in their cluster label; silhouette score should be negative; outlier_reviews should be manually inspectable |
