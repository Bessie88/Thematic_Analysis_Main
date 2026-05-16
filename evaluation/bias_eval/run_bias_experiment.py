#!/usr/bin/env python3
"""
run_bias_experiment.py
Run the full thematic analysis pipeline on pre-sampled bias scenario files.

Dataset field mappings:
  ai_healthcare  : text=text, indicator=indicator (CSV)
  ai_posts       : text=text, indicator=indicator (CSV)
  school_burnout : text=text, indicator=indicator (CSV)
  climate        : text=ad_creative_body, indicator=sampled_code (CSV)

Usage:
    python run_bias_experiment.py \
        --dataset ai_healthcare \
        --dataset-dir /scratch/yucai/data/bias_sampled/ai_healthcare \
        --output-dir /scratch/yucai/bias_experiment/ai_healthcare/qwen3-27b-fp8/run_1 \
        --model-type qwen
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATASET_FIELDS = {
    "ai_healthcare":  {"text": "text",             "indicator": "indicator"},
    "school_burnout": {"text": "text",             "indicator": "indicator"},
    "ai_posts":       {"text": "text",             "indicator": "indicator"},
    "climate":        {"text": "ad_creative_body", "indicator": "sampled_code"},
}

# Level 1 = cluster ↔ indicator_col
# Level 2 = meta_theme ↔ dimension_col (empty = skip Level 2)
DATASET_EVAL_COLS = {
    "ai_healthcare":  {"indicator_col": "indicator",    "dimension_col": "dimension"},
    "school_burnout": {"indicator_col": "indicator",    "dimension_col": "dimension"},
    "ai_posts":       {"indicator_col": "indicator",    "dimension_col": "dimension"},
    "climate":        {"indicator_col": "sampled_code", "dimension_col": ""},
}

# Extra evaluate_theme_recovery.py arguments per dataset.
# extra_indicator_cols : additional CSV columns merged into Level-1 gold set
# dim_map              : JSON string mapping Level-1 → Level-2 labels
_CLIMATE_DIM_MAP = json.dumps({
    "CA": "Community & Resilience",
    "CB": "Community & Resilience",
    "GA": "Green Innovation and Climate Solutions",
    "GC": "Green Innovation and Climate Solutions",
    "PA": "Pragmatism / Pragmatic Energy Mix",
    "PB": "Pragmatism / Pragmatic Energy Mix",
    "SA": "Patriotic Energy Mix",
})

DATASET_EVAL_EXTRA: Dict[str, Dict] = {
    "ai_healthcare":  {"extra_indicator_cols": [], "dim_map": ""},
    "school_burnout": {"extra_indicator_cols": [], "dim_map": ""},
    "ai_posts":       {"extra_indicator_cols": [], "dim_map": ""},
    "climate":        {"extra_indicator_cols": [], "dim_map": _CLIMATE_DIM_MAP},
}

# Full descriptions for gold indicator abbreviations, used for semantic similarity only.
# Other datasets use self-descriptive labels so no mapping is needed.
DATASET_IND_DESC: Dict[str, Dict[str, str]] = {
    "climate": {
        "CA": "Community contribution — oil and gas sector supports local economies, tax revenues, and community programs",
        "CB": "Jobs contribution — oil and gas sector creates and sustains employment and workers' livelihoods",
        "GA": "Green innovation — oil and gas sector reduces emissions, sets climate targets, and invests in low-carbon technology",
        "GC": "Clean fossil fuels — natural gas or lower-carbon fuels presented as climate solutions",
        "PA": "Pragmatic energy — oil and gas presented as essential, reliable, affordable, and necessary for the energy system",
        "PB": "Raw materials — fossil fuels presented as necessary inputs for plastics, medical supplies, and everyday manufactured goods",
        "SA": "Patriotic energy — domestic oil and gas production framed as beneficial for energy independence and national security",
    },
}

RESEARCH_QUESTIONS = {
    "ai_healthcare":  "What themes and concerns appear in this text about AI and healthcare?",
    "ai_posts":       "What themes and concerns appear in this text about AI and software development?",
    "climate":        "What themes and messages appear in this energy-related text?",
    "school_burnout": "What themes appear in this text about student experiences at school?",
}

SCENARIOS = ["balanced", "imbalanced", "rare_heavy"]
EMBED_MODEL = str(HERE / "agents" / "weights" / "Qwen3-Embedding-0.6B")
RANDOM_SEED = 42
PORT = int(os.environ.get("SGLANG_PORT", 8000))
MAX_RETRIES = 3


def make_llm(model_type: str = "qwen"):
    from langchain_openai import ChatOpenAI
    kwargs: dict = dict(
        model="llm",
        openai_api_key="EMPTY",
        openai_api_base=f"http://localhost:{PORT}/v1",
        temperature=0,
        max_tokens=4096,
    )
    return ChatOpenAI(**kwargs)


def _llm_call(llm, skill_key: str, prompt: str) -> str:
    from agents.core.skills import llm_invoke_with_skill
    from agents.core.utils import remove_think_tags
    return remove_think_tags(llm_invoke_with_skill(llm, skill_key, prompt))


def load_scenario_file(path: Path, text_col: str, indicator_col: str):
    """Load a CSV or JSON scenario file. Returns (pos_to_text, pos_to_indicator, indicators)."""
    pos_to_text: Dict[int, str] = {}
    pos_to_indicator: Dict[int, str] = {}

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for pos, row in enumerate(rows, start=1):
            pos_to_text[pos] = str(row.get(text_col, "")).strip()
            pos_to_indicator[pos] = str(row.get(indicator_col, "")).strip()
    else:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for pos, row in enumerate(rows, start=1):
            pos_to_text[pos] = str(row.get(text_col, "")).strip()
            pos_to_indicator[pos] = str(row.get(indicator_col, "")).strip()

    pos_to_text = {p: t for p, t in pos_to_text.items()
                   if t and pos_to_indicator.get(p)}
    pos_to_indicator = {p: v for p, v in pos_to_indicator.items() if p in pos_to_text}
    indicators = sorted(set(pos_to_indicator.values()))
    return pos_to_text, pos_to_indicator, indicators


# ── Pipeline stages ────────────────────────────────────────────────────────────

def _parse_codes(output: str) -> List[str]:
    import re
    codes = []
    for line in output.strip().splitlines():
        line = line.strip()
        m = re.match(r'^[-*\d.]+\s+(.*)', line)
        if m:
            code = m.group(1).strip().strip('"').strip("'")
            if code:
                codes.append(code)
        elif line and not line.startswith('#'):
            codes.append(line.strip('"').strip("'"))
    return [c for c in codes if c]


def _open_code_one(text: str, llm, research_question: str = "") -> List[str]:
    import time
    from agents.core.prompts import open_coding_prompt
    for attempt in range(MAX_RETRIES):
        try:
            raw = _llm_call(llm, "open_coding", open_coding_prompt(research_question, text))
            codes = _parse_codes(raw)
            if codes:
                return codes
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"      open_coding failed after {MAX_RETRIES} attempts: {e}")
    return []


def run_open_coding(sampled_ids: List[int], pos_to_text: Dict[int, str], llm, research_question: str = "") -> Dict[int, List[str]]:
    codes_by_id: Dict[int, List[str]] = {}
    total = len(sampled_ids)

    def _one(sid):
        text = pos_to_text.get(sid, "")
        return sid, _open_code_one(text, llm, research_question) if text else []

    workers = int(os.environ.get("GT_OPEN_CODING_WORKERS", "8"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, sid): sid for sid in sampled_ids}
        for fut in as_completed(futures):
            sid, codes = fut.result()
            codes_by_id[sid] = codes
            print(f"      [{len(codes_by_id)}/{total}] open coding done: {sid}", flush=True)
    return codes_by_id


def run_axial_coding(
    sampled_ids: List[int],
    codes_by_id: Dict[int, List[str]],
    embed_model,
) -> Tuple[Dict[str, List[str]], Dict[str, str], List[str], List[int]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    norm_to_canon: Dict[str, str] = {}
    for sid in sampled_ids:
        for code in codes_by_id.get(sid, []):
            n = code.strip().lower()
            if n not in norm_to_canon:
                norm_to_canon[n] = code.strip()
    all_codes = list(norm_to_canon.values())

    if len(all_codes) < 5:
        cluster_to_codes = {"0": all_codes}
        code_to_cluster = {c: "0" for c in all_codes}
        return cluster_to_codes, code_to_cluster, all_codes, [0] * len(all_codes)

    print(f"      Embedding {len(all_codes)} codes ...", flush=True)
    embs = embed_model.encode(all_codes, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)

    n = len(embs)
    K_MIN = max(3, min(5, n // 10))
    k_max = min(20, max(K_MIN + 1, n // 3))

    best_k, best_sil = K_MIN, -1.0
    for k in range(K_MIN, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        lbl = km.fit_predict(embs)
        if len(set(lbl)) < 2:
            continue
        sil = silhouette_score(embs, lbl, sample_size=min(2000, n))
        if sil > best_sil:
            best_sil, best_k = sil, k

    km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    lbl = km.fit_predict(embs)

    cluster_to_codes: Dict[str, List[str]] = defaultdict(list)
    code_to_cluster: Dict[str, str] = {}
    for code, label in zip(all_codes, lbl):
        cid = str(label)
        cluster_to_codes[cid].append(code)
        code_to_cluster[code] = cid

    for sid in sampled_ids:
        for code in codes_by_id.get(sid, []):
            n_key = code.strip().lower()
            canon = norm_to_canon.get(n_key)
            if canon and code not in code_to_cluster and canon in code_to_cluster:
                code_to_cluster[code] = code_to_cluster[canon]

    return dict(cluster_to_codes), code_to_cluster, all_codes, lbl.tolist()


def run_high_level(cluster_to_codes: Dict[str, List[str]], llm, research_question: str = "") -> Dict[str, str]:
    import re
    from agents.core.prompts import high_level_code_generation_prompt
    from agents.core.utils import clean_and_parse_json

    def _one(cid, codes):
        bulleted = "\n".join(f"- {c}" for c in codes[:40])
        raw = _llm_call(llm, "high_level_code_generation", high_level_code_generation_prompt(bulleted, research_question))
        try:
            parsed = clean_and_parse_json(raw)
        except ValueError:
            parsed = {}
        label = ""
        if isinstance(parsed, dict):
            label = parsed.get("label", parsed.get("theme", parsed.get("name", "")))
        if not label:
            m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
            if m:
                label = m.group(1)
        if not label or len(label) > 120:
            label = f"Cluster {cid}"
        return cid, label.strip()

    codebook: Dict[str, str] = {}
    workers = int(os.environ.get("GT_HIGH_LEVEL_WORKERS", "8"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, cid, codes): cid for cid, codes in cluster_to_codes.items()}
        for fut in as_completed(futures):
            cid, label = fut.result()
            codebook[cid] = label
    return codebook


def run_refine(
    cluster_to_codes: Dict[str, List[str]],
    codebook: Dict[str, str],
    code_to_cluster: Dict[str, str],
    embed_model,
    llm,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    import re
    from agents.core.prompts import refine_cluster_assignments_prompt

    cid_list = sorted(cluster_to_codes.keys(), key=int)
    labels = [codebook.get(cid, f"Cluster {cid}") for cid in cid_list]
    label_to_cids: Dict[str, List[str]] = defaultdict(list)
    for cid, lbl in codebook.items():
        label_to_cids[lbl].append(cid)

    embs = embed_model.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)
    cid_to_pos = {cid: i for i, cid in enumerate(cid_list)}

    new_ctc = {cid: list(codes) for cid, codes in cluster_to_codes.items()}
    new_c2c = dict(code_to_cluster)

    for cid in cid_list:
        codes = cluster_to_codes.get(cid, [])
        if not codes:
            continue
        label = codebook.get(cid, f"Cluster {cid}")
        pos = cid_to_pos[cid]

        sims = embs @ embs[pos]
        sims[pos] = -np.inf
        k_take = min(5, len(cid_list) - 1)
        top_idx = np.argpartition(-sims, k_take - 1)[:k_take]
        other_str = ", ".join(labels[j] for j in sorted(top_idx, key=lambda j: -sims[j]))

        MAX_CHUNK = 44
        codes_set = set(codes)
        for chunk_start in range(0, len(codes), MAX_CHUNK):
            chunk = codes[chunk_start: chunk_start + MAX_CHUNK]
            bulleted = "\n".join(f"- {c}" for c in chunk)
            raw = _llm_call(llm, "refine_cluster_assignments",
                            refine_cluster_assignments_prompt(label, bulleted, other_str))
            for line in raw.splitlines():
                m = re.search(
                    r'MOVE:\s*["\']([^"\']+)["\']\s*[→>]\s*["\']([^"\']+)["\']',
                    line.strip(), re.IGNORECASE,
                )
                if not m:
                    continue
                code, tgt_label = m.group(1).strip(), m.group(2).strip()
                if code not in codes_set:
                    continue
                tgt_cids = label_to_cids.get(tgt_label, [])
                if len(tgt_cids) != 1 or tgt_cids[0] == cid:
                    continue
                tgt_cid = tgt_cids[0]
                if code in new_ctc.get(cid, []):
                    new_ctc[cid].remove(code)
                    new_ctc.setdefault(tgt_cid, []).append(code)
                    new_c2c[code] = tgt_cid

    return new_ctc, new_c2c


def run_meta_theme(codebook: Dict[str, str], llm, research_question: str) -> List[Dict]:
    from agents.core.prompts import meta_theme_grouping_prompt
    from agents.core.utils import clean_and_parse_json
    from agents.core.pipeline_helpers import normalize_meta_theme_count

    labels_json = json.dumps(codebook, indent=2)
    raw = _llm_call(llm, "meta_theme_grouping",
                    meta_theme_grouping_prompt(labels_json, research_question))
    try:
        parsed = clean_and_parse_json(raw)
    except ValueError:
        parsed = {}
    meta_themes = [m for m in parsed.get("meta_themes", []) if isinstance(m, dict)]

    all_cids = set(codebook.keys())
    assigned = {str(cid) for mt in meta_themes for cid in mt.get("cluster_ids", [])}
    missing = all_cids - assigned
    if missing and meta_themes:
        largest = max(meta_themes, key=lambda m: len(m.get("cluster_ids", [])))
        for cid in missing:
            largest["cluster_ids"].append(cid)

    return normalize_meta_theme_count(meta_themes, len(all_cids))


def match_clusters_to_indicators(
    sampled_ids: List[int],
    codes_by_id: Dict[int, List[str]],
    code_to_cluster: Dict[str, str],
    pos_to_text: Dict[int, str],
    pos_to_indicator: Dict[int, str],
    indicators: List[str],
    embed_model,
) -> Dict[str, str]:
    all_texts = [pos_to_text[sid] for sid in sampled_ids]
    text_embs = embed_model.encode(all_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    text_embs = np.asarray(text_embs, dtype=np.float32)
    sid_to_pos = {sid: i for i, sid in enumerate(sampled_ids)}

    def centroid(sids_iter) -> Optional[np.ndarray]:
        vecs = [text_embs[sid_to_pos[s]] for s in sids_iter if s in sid_to_pos]
        if not vecs:
            return None
        v = np.mean(vecs, axis=0)
        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)

    cluster_ids = set(code_to_cluster.values())
    cluster_centroids: Dict[str, np.ndarray] = {}
    for cid in cluster_ids:
        members = [sid for sid in sampled_ids
                   if any(code_to_cluster.get(c) == cid for c in codes_by_id.get(sid, []))]
        c = centroid(members)
        if c is not None:
            cluster_centroids[cid] = c

    indicator_centroids: Dict[str, np.ndarray] = {}
    for ind in indicators:
        members = [sid for sid in sampled_ids if pos_to_indicator.get(sid) == ind]
        c = centroid(members)
        if c is not None:
            indicator_centroids[ind] = c

    cluster_to_indicator: Dict[str, str] = {}
    for cid, cc in cluster_centroids.items():
        if not indicator_centroids:
            break
        best_ind = max(indicator_centroids, key=lambda ind: float(np.dot(cc, indicator_centroids[ind])))
        cluster_to_indicator[cid] = best_ind

    return cluster_to_indicator


def compute_bias_metrics(
    sampled_ids: List[int],
    codes_by_id: Dict[int, List[str]],
    code_to_cluster: Dict[str, str],
    cluster_to_indicator: Dict[str, str],
    pos_to_indicator: Dict[int, str],
    indicators: List[str],
) -> Dict:
    pred: Dict[int, Optional[str]] = {}
    for sid in sampled_ids:
        counts = Counter(
            code_to_cluster.get(code)
            for code in codes_by_id.get(sid, [])
            if code_to_cluster.get(code) is not None
        )
        if counts:
            maj_cid = counts.most_common(1)[0][0]
            pred[sid] = cluster_to_indicator.get(str(maj_cid))
        else:
            pred[sid] = None

    input_counts = Counter(pos_to_indicator[sid] for sid in sampled_ids)
    total_in = sum(input_counts.values())
    input_props = {ind: input_counts.get(ind, 0) / total_in for ind in indicators}

    output_counts = Counter(pred[sid] for sid in sampled_ids if pred[sid])
    total_out = sum(output_counts.values())
    output_props = {ind: output_counts.get(ind, 0) / max(total_out, 1) for ind in indicators}

    per_indicator: Dict[str, Dict] = {}
    for ind in indicators:
        n_in = input_counts.get(ind, 0)
        n_correct = sum(1 for sid in sampled_ids
                        if pos_to_indicator.get(sid) == ind and pred.get(sid) == ind)
        per_indicator[ind] = {
            "input_count":       n_in,
            "input_proportion":  round(input_props[ind], 4),
            "output_proportion": round(output_props[ind], 4),
            "recall":            round(n_correct / n_in, 4) if n_in > 0 else 0.0,
        }

    x = np.array([input_props[ind] for ind in indicators])
    y = np.array([output_props[ind] for ind in indicators])
    mae = float(np.mean(np.abs(x - y)))
    if x.std() > 1e-9 and y.std() > 1e-9:
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = 1.0 if np.allclose(x, y) else 0.0

    n_uncoded = sum(1 for sid in sampled_ids if not codes_by_id.get(sid))

    return {
        "n_sampled":            len(sampled_ids),
        "n_uncoded":            n_uncoded,
        "proportion_mae":       round(mae, 4),
        "proportion_pearson_r": round(corr, 4),
        "per_indicator":        per_indicator,
    }


# ── Scenario runner ────────────────────────────────────────────────────────────

def make_eval_gold_csv(
    scenario_file: Path,
    text_col: str,
    indicator_col: str,
    dimension_col: str,
    dataset: str,
    output_path: Path,
    extra_indicator_cols: Optional[List[str]] = None,
) -> None:
    """Create a standardised gold CSV for evaluate_theme_recovery.py.

    The primary indicator column is always written as 'indicator'.
    Any extra_indicator_cols are written with their original names so that
    evaluate_theme_recovery.py can read them via --extra-indicator-col.
    """
    extra_cols = extra_indicator_cols or []
    fieldnames = ["text", "indicator", "dimension"] + extra_cols

    rows_out = []
    with open(scenario_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = str(row.get(text_col, "")).strip()
            ind  = str(row.get(indicator_col, "")).strip()
            dim  = str(row.get(dimension_col, "")).strip() if dimension_col else ""
            if text and ind:
                out_row = {"text": text, "indicator": ind, "dimension": dim}
                for col in extra_cols:
                    out_row[col] = str(row.get(col, "")).strip()
                rows_out.append(out_row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def run_scenario(
    scenario_name: str,
    scenario_file: Path,
    text_col: str,
    indicator_col: str,
    dimension_col: str,
    dataset: str,
    output_dir: Path,
    embed_model,
    llm,
    research_question: str,
    extra_eval: Optional[Dict] = None,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*60}")

    pos_to_text, pos_to_indicator, indicators = load_scenario_file(
        scenario_file, text_col, indicator_col)
    sampled_ids = list(pos_to_text.keys())
    print(f"  Loaded {len(sampled_ids)} reviews, {len(indicators)} indicators")

    if len(sampled_ids) < 10:
        print("  Too few samples, skipping.")
        return {}

    # Stage 1: Open coding
    print(f"  [open_coding] running LLM ...")
    codes_by_id = run_open_coding(sampled_ids, pos_to_text, llm, research_question)
    n_coded = sum(1 for c in codes_by_id.values() if c)
    print(f"  [open_coding] {n_coded}/{len(sampled_ids)} produced codes")
    (output_dir / f"codes_{scenario_name}.json").write_text(
        json.dumps({"scenario": scenario_name,
                    "codes_per_review": [[sid, codes_by_id[sid]] for sid in sampled_ids]},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # Stage 2: Axial coding
    print(f"  [axial_coding] embedding + K-means ...")
    cluster_to_codes, code_to_cluster, ax_all_codes, ax_labels = run_axial_coding(
        sampled_ids, codes_by_id, embed_model)
    print(f"  [axial_coding] {len(cluster_to_codes)} clusters")
    # Save in format compatible with evaluate_theme_recovery.py
    (output_dir / f"clustered_{scenario_name}.json").write_text(
        json.dumps({"scenario":         scenario_name,
                    "all_codes":        ax_all_codes,
                    "labels":           ax_labels,
                    "k":                len(cluster_to_codes),
                    "cluster_to_codes": cluster_to_codes,
                    "code_to_cluster":  code_to_cluster,
                    "codes_per_review": [[sid, codes_by_id[sid]] for sid in sampled_ids]},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # Stage 3: High-level labeling
    print(f"  [high_level] labeling clusters ...")
    codebook = run_high_level(cluster_to_codes, llm, research_question)
    (output_dir / f"codebook_{scenario_name}.json").write_text(
        json.dumps({"scenario": scenario_name, "codebook": codebook},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # Stage 4: Refine
    print(f"  [refine] adjusting cluster assignments ...")
    cluster_to_codes_r, code_to_cluster_r = run_refine(
        cluster_to_codes, codebook, code_to_cluster, embed_model, llm)
    (output_dir / f"refined_{scenario_name}.json").write_text(
        json.dumps({"scenario": scenario_name,
                    "cluster_to_codes_refined": cluster_to_codes_r,
                    "code_to_cluster_refined": code_to_cluster_r},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # Stage 5: Meta-theme
    print(f"  [meta_theme] grouping clusters ...")
    meta_themes = run_meta_theme(codebook, llm, research_question)
    (output_dir / f"meta_themes_{scenario_name}.json").write_text(
        json.dumps({"scenario": scenario_name, "meta_themes": meta_themes},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # Eval
    print(f"  [eval] centroid matching + bias metrics ...")
    cti = match_clusters_to_indicators(
        sampled_ids, codes_by_id, code_to_cluster_r,
        pos_to_text, pos_to_indicator, indicators, embed_model)
    metrics = compute_bias_metrics(
        sampled_ids, codes_by_id, code_to_cluster_r,
        cti, pos_to_indicator, indicators)

    print(f"  [eval] MAE={metrics['proportion_mae']:.4f}  Pearson r={metrics['proportion_pearson_r']:.4f}")

    # ── Run evaluate_theme_recovery.py for this scenario ──────────────────────
    print(f"  [theme_recovery] running evaluation ...")
    eval_gold_csv = output_dir / f"gold_{scenario_name}.csv"
    make_eval_gold_csv(scenario_file, text_col, indicator_col, dimension_col,
                       dataset, eval_gold_csv,
                       extra_indicator_cols=(extra_eval or {}).get("extra_indicator_cols"))

    eval_results_path = output_dir / f"theme_recovery_{scenario_name}.json"
    eval_report_path  = output_dir / f"theme_recovery_report_{scenario_name}.html"

    eval_cmd = [
        sys.executable,
        str(HERE / "evaluate_theme_recovery.py"),
        "--gold-csv",      str(eval_gold_csv),
        "--clustered",     str(output_dir / f"clustered_{scenario_name}.json"),
        "--codebook",      str(output_dir / f"codebook_{scenario_name}.json"),
        "--meta-themes",   str(output_dir / f"meta_themes_{scenario_name}.json"),
        "--output",        str(eval_results_path),
        "--indicator-col", "indicator",
        "--dimension-col", dimension_col,
    ]
    extra = extra_eval or {}
    if extra.get("extra_indicator_cols"):
        eval_cmd += ["--extra-indicator-col", ",".join(extra["extra_indicator_cols"])]
    if extra.get("dim_map"):
        eval_cmd += ["--dim-map", extra["dim_map"]]
    import subprocess
    try:
        subprocess.run(eval_cmd, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        print(f"  [theme_recovery] results saved: {eval_results_path}")

        report_cmd = [
            sys.executable,
            str(HERE / "generate_report.py"),
            "--results",   str(eval_results_path),
            "--output",    str(eval_report_path),
            "--skip-eval",
        ]
        subprocess.run(report_cmd, check=True)
        print(f"  [theme_recovery] report saved: {eval_report_path}")
    except Exception as e:
        print(f"  WARNING: evaluation failed: {e}")

    input_props = {
        ind: round(sum(1 for sid in sampled_ids if pos_to_indicator[sid] == ind) / len(sampled_ids), 4)
        for ind in indicators
    }

    return {
        "n_sampled":            len(sampled_ids),
        "indicators":           indicators,
        "input_proportions":    input_props,
        "metrics":              metrics,
        "meta_themes":          meta_themes,
        "codebook":             codebook,
        "cluster_to_indicator": cti,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Run bias experiment on pre-sampled scenario files")
    p.add_argument("--dataset",     required=True, choices=list(DATASET_FIELDS.keys()))
    p.add_argument("--dataset-dir", required=True, help="Directory with scenario CSV/JSON files")
    p.add_argument("--output-dir",  required=True, help="Output directory (must not already contain results)")
    p.add_argument("--model-type",  default="qwen", help="'qwen' or 'gemma' — affects LLM chat kwargs")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fields            = DATASET_FIELDS[args.dataset]
    eval_cols         = DATASET_EVAL_COLS[args.dataset]
    research_question = RESEARCH_QUESTIONS[args.dataset]

    print(f"Dataset  : {args.dataset}")
    print(f"RQ       : {research_question}")
    print(f"Output   : {output_dir}")
    print(f"LLM type : {args.model_type}")

    ext = ".csv"
    scenario_files: Dict[str, Path] = {}
    for scenario in SCENARIOS:
        path = dataset_dir / f"{scenario}{ext}"
        if path.exists():
            scenario_files[scenario] = path
        else:
            print(f"WARNING: {path} not found, skipping.")

    if not scenario_files:
        print("ERROR: No scenario files found.")
        sys.exit(1)

    print(f"Scenarios: {list(scenario_files.keys())}")

    print("\nLoading embedding model ...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")

    print("Connecting to LLM (SGLang) ...")
    llm = make_llm(args.model_type)

    extra_eval = DATASET_EVAL_EXTRA[args.dataset]

    all_results: Dict[str, Dict] = {}
    for scenario_name, scenario_file in scenario_files.items():
        try:
            result = run_scenario(
                scenario_name, scenario_file,
                fields["text"], fields["indicator"],
                eval_cols["dimension_col"],
                args.dataset,
                output_dir, embed_model, llm, research_question,
                extra_eval=extra_eval,
            )
            if result:
                all_results[scenario_name] = result
        except Exception as e:
            import traceback
            print(f"ERROR in scenario {scenario_name}: {e}")
            traceback.print_exc()

    results_path = output_dir / "bias_results.json"
    results_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAll results saved to: {results_path}")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Scenario':<20} {'MAE':>8} {'Pearson r':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    for name, res in all_results.items():
        m = res["metrics"]
        print(f"  {name:<20} {m['proportion_mae']:>8.4f} {m['proportion_pearson_r']:>10.4f}")


if __name__ == "__main__":
    main()
