#!/usr/bin/env python3
"""
Theme Recovery Evaluation
=========================
Evaluates pipeline output against gold labels at TWO levels:

  Level 1 (fine):   cluster    ↔ indicator  (37+ clusters  vs 9  gold indicators)
  Level 2 (coarse): meta_theme ↔ dimension  (5 meta_themes vs 3  gold dimensions)

Both levels use embedding-only centroid matching (no LLM, no threshold).

Usage:
    python evaluate_theme_recovery.py \
        --gold-csv data/prepared/school_burnout_synthetic.csv \
        --output results.json
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise ImportError("scikit-learn is required: pip install scikit-learn")

try:
    from rouge_score import rouge_scorer as rouge_scorer_lib
except ImportError:
    raise ImportError("rouge-score is required: pip install rouge-score")

try:
    from bert_score import score as bert_score_fn
except ImportError:
    raise ImportError("bert-score is required: pip install bert-score")

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_GOLD_CSV    = os.path.join(HERE, "data/prepared/school_burnout_synthetic.csv")
DEFAULT_CLUSTERED   = os.path.join(HERE, "agents/outputs/data/gt_clustered_codes.json")
DEFAULT_CODEBOOK    = os.path.join(HERE, "agents/outputs/data/codebook.json")
DEFAULT_META_THEMES = os.path.join(HERE, "agents/outputs/data/gt_meta_themes.json")
EMBEDDING_MODEL     = os.path.join(HERE, "agents/weights/Qwen3-Embedding-0.6B")


# ── 1. Data loading ────────────────────────────────────────────────────────────

def load_gold_level(
    csv_path: str,
    level: str,
    text_col: str = "text",
    extra_cols: Optional[List[str]] = None,
) -> Tuple[Dict[int, Set[str]], List[str], Dict[int, str]]:
    """
    Returns:
        gold_sets:   {1-indexed sid -> {label(s) at given level}}
        canonical:   sorted unique gold labels
        id_to_text:  {1-indexed sid -> sentence text}

    extra_cols: additional CSV columns whose values are merged into each row's
        gold set (for multi-label evaluation, e.g. ESconv seeker where both
        emotion type and problem type contribute to the same fine-grained level).
    """
    gold_sets:  Dict[int, Set[str]] = {}
    id_to_text: Dict[int, str]      = {}
    all_cols = [level] + (extra_cols or [])
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            sid = i + 1  # 1-indexed to match codes_per_review convention
            labels = {row[c] for c in all_cols if row.get(c, "").strip()}
            gold_sets[sid]  = labels if labels else {""}
            id_to_text[sid] = row.get(text_col, "")
    canonical = sorted({lbl for s in gold_sets.values() for lbl in s})
    return gold_sets, canonical, id_to_text


def derive_dim_gold(
    gold_sets_ind: Dict[int, Set[str]],
    dim_map: Dict[str, str],
) -> Tuple[Dict[int, Set[str]], List[str]]:
    """
    Derive coarse-level gold sets by mapping indicator labels through dim_map.
    Used for datasets (e.g., climate) where Level-2 labels are determined by
    a static code→super-category mapping rather than a separate CSV column.
    Labels absent from dim_map pass through unchanged.
    """
    gold_sets_dim: Dict[int, Set[str]] = {}
    for sid, ind_labels in gold_sets_ind.items():
        gold_sets_dim[sid] = {dim_map.get(lbl, lbl) for lbl in ind_labels}
    canonical = sorted({lbl for s in gold_sets_dim.values() for lbl in s})
    return gold_sets_dim, canonical


def load_clustered(
    clustered_path: str,
) -> Tuple[Dict[int, List[str]], Dict[str, str]]:
    d = json.load(open(clustered_path, encoding="utf-8"))
    # codes_per_review: [[sid, [code, ...]], ...] → {sid -> [code, ...]}
    codes_per_review = {item[0]: item[1] for item in d["codes_per_review"]}
    all_codes: List[str] = d["all_codes"]
    labels:    List[int] = d["labels"]  # parallel array: labels[i] = cluster id of all_codes[i]
    code_to_cluster = {code: str(cid) for code, cid in zip(all_codes, labels)}
    return codes_per_review, code_to_cluster


def load_cluster_labels(codebook_path: str) -> Dict[str, str]:
    cb = json.load(open(codebook_path, encoding="utf-8"))
    return cb["codebook"]  # {cluster_id -> human-readable label}


def load_meta_themes(
    meta_themes_path: str,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
        meta_theme_labels:      {mt_id -> name}
        meta_theme_to_clusters: {mt_id -> [cluster_ids]}
    """
    d = json.load(open(meta_themes_path, encoding="utf-8"))
    labels: Dict[str, str]       = {}
    mt2c:   Dict[str, List[str]] = {}
    for i, mt in enumerate(d["meta_themes"]):
        mt_id = str(i)  # assign stable string IDs from list position (JSON has no native IDs)
        labels[mt_id] = mt["name"]
        mt2c[mt_id]   = mt["cluster_ids"]
    return labels, mt2c


# ── 2. Build coverage ──────────────────────────────────────────────────────────

def build_cluster_to_sids(
    codes_per_review: Dict[int, List[str]],
    code_to_cluster:  Dict[str, str],
) -> Dict[str, Set[int]]:
    # For each cluster: collect the set of review IDs that have at least one code in it.
    # Used later to compute the cluster's centroid embedding from its covered reviews.
    c2s: Dict[str, Set[int]] = {}
    for sid, codes in codes_per_review.items():
        for code in codes:
            cid = code_to_cluster.get(code)
            if cid is not None:
                c2s.setdefault(cid, set()).add(sid)
    return c2s


def build_meta_theme_to_sids(
    meta_theme_to_clusters: Dict[str, List[str]],
    cluster_to_sids:        Dict[str, Set[int]],
) -> Dict[str, Set[int]]:
    # A meta-theme covers the union of all reviews from its constituent clusters.
    mt2s: Dict[str, Set[int]] = {}
    for mt_id, cluster_ids in meta_theme_to_clusters.items():
        sids: Set[int] = set()
        for cid in cluster_ids:
            sids |= cluster_to_sids.get(cid, set())
        mt2s[mt_id] = sids
    return mt2s


# ── 3. Embeddings ─────────────────────────────────────────────────────────────

def embed(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    # normalize_embeddings=True makes cosine similarity equivalent to dot product,
    # which is cheaper and numerically stable for the matrix multiplications below.
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ── 4. Entity majority gold ────────────────────────────────────────────────────

def entity_majority_gold(
    sids:      Set[int],
    gold_sets: Dict[int, Set[str]],
) -> Optional[str]:
    # Returns the gold label most common among reviews covered by this cluster/meta-theme.
    # Used as the "true answer" when checking whether centroid top-1 is correct.
    counter: Counter = Counter()
    for sid in sids:
        for label in gold_sets.get(sid, set()):
            counter[label] += 1
    return max(counter, key=counter.__getitem__) if counter else None


# ── 5. Mapping dataclass ──────────────────────────────────────────────────────

@dataclass
class MappingResult:
    cluster_id:        str
    cluster_label:     str
    top1_gold:         str           # gold label with highest centroid cosine similarity
    top1_similarity:   float
    majority_gold:     Optional[str] # ground-truth label by majority vote; None if cluster is empty
    is_top1_correct:   Optional[bool]  # None when majority_gold is None (no reviews → no ground truth)
    ranked_candidates: List[Tuple[str, float]]  # all gold labels sorted by cosine similarity descending


# ── 6. Centroid-based mapping ─────────────────────────────────────────────────

def map_entities(
    entity_labels:  Dict[str, str],
    entity_to_sids: Dict[str, Set[int]],
    gold_sets:      Dict[int, Set[str]],
    canonical:      List[str],
    emb_model:      SentenceTransformer,
    id_to_text:     Dict[int, str],
    sent_embs:      np.ndarray,
    unique_texts:   List[str],
    text2idx:       Dict[str, int],
) -> Tuple[Dict[str, MappingResult], List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Centroid-based matching of entities (clusters or meta_themes)
    against gold labels using pre-computed sentence embeddings.

    Returns:
        mapping_results:  {entity_id -> MappingResult}
        entity_ids:       entity IDs in row order of sim_matrix
        sim_matrix:       (n_entities, n_canonical) centroid cosine similarity
        label_sim_matrix: (n_entities, n_canonical) label-string cosine similarity
        canon_embs:       (n_canonical, dim) gold label-string embeddings
    """
    entity_ids = sorted(entity_labels.keys(), key=lambda x: int(x))

    def centroid(sids_iter, fallback: str) -> np.ndarray:
        texts = [id_to_text[s] for s in sids_iter if s in id_to_text and id_to_text[s]]
        if texts:
            # Look up pre-computed embeddings by text to avoid re-encoding
            vecs = sent_embs[[text2idx[t] for t in texts]]
            c    = vecs.mean(axis=0)
        else:
            # Empty cluster (no reviews cover it): fall back to embedding the label string
            c = embed([fallback], emb_model)[0]
        norm = float(np.linalg.norm(c))
        return c / norm if norm > 1e-9 else c

    # Gold centroids (n_canonical, dim)
    gold_sids_map: Dict[str, List[int]] = defaultdict(list)
    for sid, labels in gold_sets.items():
        for lbl in labels:
            gold_sids_map[lbl].append(sid)

    gold_centroids = np.stack(
        [centroid(gold_sids_map.get(lbl, []), lbl) for lbl in canonical]
    )

    # Entity centroids (n_entities, dim)
    entity_centroids = np.stack([
        centroid(entity_to_sids.get(eid, set()), entity_labels[eid])
        for eid in entity_ids
    ])

    sim_matrix = entity_centroids @ gold_centroids.T  # (n_entities, n_canonical)

    # Label-string similarity: how well does the cluster's *name* alone match gold?
    # This is a baseline to compare against centroid similarity — if label_sim ≈ centroid_sim,
    # the naming did the work; if centroid_sim >> label_sim, the review content helped.
    canon_embs        = embed(canonical, emb_model)
    entity_label_embs = embed([entity_labels[eid] for eid in entity_ids], emb_model)
    label_sim_matrix  = entity_label_embs @ canon_embs.T

    results: Dict[str, MappingResult] = {}
    for i, eid in enumerate(entity_ids):
        label = entity_labels[eid]
        sims  = sim_matrix[i]

        ranked_idx        = np.argsort(sims)[::-1]
        ranked_candidates = [(canonical[j], float(sims[j])) for j in ranked_idx]

        top1_gold  = ranked_candidates[0][0]
        top1_sim   = ranked_candidates[0][1]
        sids_      = entity_to_sids.get(eid, set())
        majority   = entity_majority_gold(sids_, gold_sets)
        is_correct = (top1_gold == majority) if majority is not None else None

        results[eid] = MappingResult(
            cluster_id=eid,
            cluster_label=label,
            top1_gold=top1_gold,
            top1_similarity=round(top1_sim, 6),
            majority_gold=majority,
            is_top1_correct=is_correct,
            ranked_candidates=ranked_candidates,
        )

    return results, entity_ids, sim_matrix, label_sim_matrix, canon_embs


# ── 7. Build predicted sets ───────────────────────────────────────────────────

def build_pred_sets_cluster(
    codes_per_review: Dict[int, List[str]],
    code_to_cluster:  Dict[str, str],
    mapping_results:  Dict[str, MappingResult],
) -> Dict[int, Set[str]]:
    """Sentence-level predictions via clusters (Level 1)."""
    pred: Dict[int, Set[str]] = {}
    for sid, codes in codes_per_review.items():
        best_mr: Optional[MappingResult] = None
        for code in codes:
            cid = code_to_cluster.get(code)
            if cid is None:
                continue
            mr = mapping_results.get(cid)
            if mr is None:
                continue
            # A review may touch multiple clusters; pick the one with highest centroid similarity
            # as the dominant signal for that review's predicted indicator
            if best_mr is None or mr.top1_similarity > best_mr.top1_similarity:
                best_mr = mr
        pred[sid] = {best_mr.top1_gold} if best_mr is not None else set()
    return pred


def build_pred_sets_meta(
    codes_per_review: Dict[int, List[str]],
    code_to_cluster:  Dict[str, str],
    cluster_to_meta:  Dict[str, str],
    mapping_results:  Dict[str, MappingResult],
) -> Dict[int, Set[str]]:
    """Sentence-level predictions via meta_themes (Level 2).
    Same winner-takes-all logic as Level 1, but traverses code → cluster → meta_theme.
    """
    pred: Dict[int, Set[str]] = {}
    for sid, codes in codes_per_review.items():
        best_mr: Optional[MappingResult] = None
        for code in codes:
            cid   = code_to_cluster.get(code)
            if cid is None:
                continue
            mt_id = cluster_to_meta.get(cid)
            if mt_id is None:
                continue
            mr = mapping_results.get(mt_id)
            if mr is None:
                continue
            if best_mr is None or mr.top1_similarity > best_mr.top1_similarity:
                best_mr = mr
        pred[sid] = {best_mr.top1_gold} if best_mr is not None else set()
    return pred


# ── 8. Sentence-level set metrics ─────────────────────────────────────────────

def set_metrics(
    gold: Set[str], pred: Set[str]
) -> Tuple[float, float, float, float]:
    # Both empty counts as perfect agreement; one empty is total failure.
    if not gold and not pred:
        return 1.0, 1.0, 1.0, 1.0
    if not gold or not pred:
        return 0.0, 0.0, 0.0, 0.0
    tp        = len(gold & pred)
    precision = tp / len(pred)
    recall    = tp / len(gold)
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    jaccard   = tp / len(gold | pred)
    return precision, recall, f1, jaccard


def compute_global_metrics(
    gold_sets: Dict[int, Set[str]],
    pred_sets: Dict[int, Set[str]],
) -> Tuple[float, float, float, float]:
    # Macro-average over all sentences: each sentence contributes equally regardless of label frequency.
    rows = [set_metrics(gold_sets[sid], pred_sets.get(sid, set()))
            for sid in sorted(gold_sets.keys())]
    return (
        float(np.mean([r[0] for r in rows])),
        float(np.mean([r[1] for r in rows])),
        float(np.mean([r[2] for r in rows])),
        float(np.mean([r[3] for r in rows])),
    )


def compute_mean_labelwise_kappa(
    gold_sets: Dict[int, Set[str]],
    pred_sets: Dict[int, Set[str]],
    canonical: List[str],
) -> float:
    # Treat each theme as a binary classifier; average Cohen's kappa across all themes.
    # More robust than global accuracy because it accounts for class imbalance.
    sids = sorted(gold_sets.keys())
    N    = len(sids)
    kappas = []
    for theme in canonical:
        g = [1 if theme in gold_sets[s] else 0 for s in sids]
        p = [1 if theme in pred_sets.get(s, set()) else 0 for s in sids]
        # Confusion matrix cells: a=TP, b=FN, c=FP, d=TN
        a = sum(gi == 1 and pi == 1 for gi, pi in zip(g, p))
        b = sum(gi == 1 and pi == 0 for gi, pi in zip(g, p))
        c = sum(gi == 0 and pi == 1 for gi, pi in zip(g, p))
        d = sum(gi == 0 and pi == 0 for gi, pi in zip(g, p))
        Po = (a + d) / N          # observed agreement
        Pe = ((a + b) / N) * ((a + c) / N) + ((c + d) / N) * ((b + d) / N)  # chance agreement
        kappas.append((Po - Pe) / (1 - Pe) if 1 - Pe > 1e-9 else (1.0 if Po >= 1.0 else 0.0))
    return float(np.mean(kappas)) if kappas else 0.0


# ── 9. Matching metrics ────────────────────────────────────────────────────────

def compute_top1_accuracy(mapping_results: Dict[str, MappingResult]) -> float:
    # Fraction of clusters whose centroid top-1 gold match equals the cluster's majority gold label.
    # Clusters with no reviews (majority_gold=None) are excluded from the denominator.
    correct = sum(1 for mr in mapping_results.values() if mr.is_top1_correct)
    total   = sum(1 for mr in mapping_results.values() if mr.is_top1_correct is not None)
    return correct / total if total > 0 else 0.0


def compute_soft_accuracy(
    mapping_results: Dict[str, MappingResult],
    canonical:       List[str],
    canon_embs:      np.ndarray,
) -> float:
    # Instead of binary correct/wrong, measure semantic closeness between predicted and true gold label.
    # A wrong prediction that maps to a semantically similar indicator still scores close to 1.0.
    label_to_idx = {label: i for i, label in enumerate(canonical)}
    scores = []
    for mr in mapping_results.values():
        if mr.majority_gold is None:
            continue
        i = label_to_idx[mr.top1_gold]
        j = label_to_idx[mr.majority_gold]
        scores.append(float(canon_embs[i] @ canon_embs[j]))
    return float(np.mean(scores)) if scores else 0.0


def compute_pairwise_auc(
    mapping_results: Dict[str, MappingResult],
    canonical:       List[str],
) -> Optional[float]:
    # For each cluster, treat its centroid similarity scores against all gold labels as a ranking problem.
    # AUC measures whether the true gold label tends to have higher similarity than the wrong ones.
    # Returns None if all clusters happen to share the same majority gold (degenerate single-class case).
    scores_list: List[float] = []
    labels_list: List[int]   = []
    for mr in mapping_results.values():
        if mr.majority_gold is None:
            continue
        sim_map = dict(mr.ranked_candidates)
        for gold_label in canonical:
            scores_list.append(sim_map.get(gold_label, 0.0))
            labels_list.append(1 if gold_label == mr.majority_gold else 0)
    if not labels_list or len(set(labels_list)) < 2:
        return None
    try:
        return float(roc_auc_score(labels_list, scores_list))
    except Exception:
        return None


def compute_rougeL(mapping_results: Dict[str, MappingResult]) -> float:
    # Compares cluster label string against its matched gold indicator string using ROUGE-L.
    # Measures lexical overlap between the pipeline's theme names and the gold vocabulary.
    scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(mr.top1_gold, mr.cluster_label)["rougeL"].fmeasure
        for mr in mapping_results.values()
    ]
    return float(np.mean(scores)) if scores else 0.0


def compute_bertscore_f1(mapping_results: Dict[str, MappingResult]) -> float:
    # BERTScore F1: semantic similarity between cluster label and matched gold label using contextual embeddings.
    # Complements ROUGE-L by capturing paraphrase similarity beyond surface word overlap.
    predictions = [mr.cluster_label for mr in mapping_results.values()]
    references  = [mr.top1_gold     for mr in mapping_results.values()]
    _, _, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return float(F1.mean().item())


# ── 10. Run one evaluation level ──────────────────────────────────────────────

def run_level(
    level_name:      str,
    entity_labels:   Dict[str, str],
    entity_to_sids:  Dict[str, Set[int]],
    gold_sets:       Dict[int, Set[str]],
    canonical:       List[str],
    id_to_text:      Dict[int, str],
    emb_model:       SentenceTransformer,
    sent_embs:       np.ndarray,
    unique_texts:    List[str],
    text2idx:        Dict[str, int],
    pred_sets_fn:    Callable[[Dict[str, MappingResult]], Dict[int, Set[str]]],
    embedding_model_path: str,
) -> dict:
    """
    Run full evaluation for one level.
    pred_sets_fn receives mapping_results and returns {sid -> Set[predicted_label]}.
    """
    print(f"\n  Building centroid similarity matrix ...")
    mapping_results, entity_ids, sim_matrix, label_sim_matrix, canon_embs = map_entities(
        entity_labels, entity_to_sids, gold_sets, canonical, emb_model,
        id_to_text, sent_embs, unique_texts, text2idx,
    )

    # Per-entity summary
    print(f"  Per-entity details ({len(mapping_results)} entities):")
    for eid in entity_ids:
        mr   = mapping_results[eid]
        flag = ("✓" if mr.is_top1_correct else "✗") if mr.is_top1_correct is not None else "—"
        print(
            f"    [{flag}] {mr.cluster_label!r:50s}"
            f"  →  {mr.top1_gold!r}  (sim={mr.top1_similarity:.4f})"
            f"  |  majority: {mr.majority_gold!r}"
        )

    # pred_sets_fn is injected so the same run_level() works for both levels
    # without needing to know whether entities are clusters or meta-themes
    pred_sets = pred_sets_fn(mapping_results)

    # ── Matching metrics ──────────────────────────────────────────────────────
    top1_acc = compute_top1_accuracy(mapping_results)

    # Construct dummy MappingResults using label-string similarity to get the label-only baseline accuracy.
    # Reuses compute_top1_accuracy without duplicating its logic.
    label_results_tmp: Dict[str, MappingResult] = {}
    for i, eid in enumerate(entity_ids):
        lsims    = label_sim_matrix[i]
        ranked_l = [canonical[j] for j in np.argsort(lsims)[::-1]]
        mg       = mapping_results[eid].majority_gold
        label_results_tmp[eid] = MappingResult(
            cluster_id=eid, cluster_label="", top1_gold=ranked_l[0],
            top1_similarity=0, majority_gold=mg,
            is_top1_correct=(ranked_l[0] == mg) if mg is not None else None,
            ranked_candidates=[],
        )
    label_top1_acc = compute_top1_accuracy(label_results_tmp)
    soft_acc       = compute_soft_accuracy(mapping_results, canonical, canon_embs)
    pairwise_auc   = compute_pairwise_auc(mapping_results, canonical)
    mean_rougeL    = compute_rougeL(mapping_results)
    print(f"  Computing BERTScore ...")
    mean_bs_f1     = compute_bertscore_f1(mapping_results)

    # ── Sentence-level metrics ─────────────────────────────────────────────────
    # Each review has exactly one gold label and one predicted label, so precision = recall = F1 = exact match.
    # We use precision (mp) as sent_acc; recall/f1/jaccard are equivalent here and discarded.
    mp, _r, _f, _j = compute_global_metrics(gold_sets, pred_sets)
    sent_acc = mp
    kappa    = compute_mean_labelwise_kappa(gold_sets, pred_sets, canonical)

    print(f"  Top-1 Accuracy (centroid)     : {top1_acc:.4f}")
    print(f"  Top-1 Accuracy (label string) : {label_top1_acc:.4f}")
    print(f"  Soft Accuracy                 : {soft_acc:.4f}")
    if pairwise_auc is not None:
        print(f"  Pairwise AUC                  : {pairwise_auc:.4f}")
    else:
        print(f"  Pairwise AUC                  : N/A (single class)")
    print(f"  Mean ROUGE-L                  : {mean_rougeL:.4f}")
    print(f"  Mean BERTScore F1             : {mean_bs_f1:.4f}")
    print(f"  Sentence Accuracy             : {sent_acc:.4f}")
    print(f"  Mean Labelwise Kappa          : {kappa:.4f}")

    # Sentence result rows: correct = True when predicted label exactly matches gold label
    sentence_rows = []
    for sid in sorted(gold_sets.keys()):
        gold = gold_sets[sid]
        pred = pred_sets.get(sid, set())
        p, _r2, _f2, _j2 = set_metrics(gold, pred)
        sentence_rows.append({
            "sentence_id": sid,
            "gold_set":    sorted(gold),
            "pred_set":    sorted(pred),
            "correct":     bool(p == 1.0),
        })

    return {
        "config": {
            "level":            level_name,
            "canonical_themes": canonical,
            "embedding_model":  embedding_model_path,
        },
        "summary": {
            "top1_accuracy_centroid":     round(top1_acc,        4),
            "top1_accuracy_label_string": round(label_top1_acc,  4),
            "soft_accuracy":              round(soft_acc,         4),
            "pairwise_auc":      round(pairwise_auc, 4) if pairwise_auc is not None else None,
            "mean_rougeL":       round(mean_rougeL,  4),
            "mean_bertscore_f1": round(mean_bs_f1,   4),
            "sentence_accuracy":    round(sent_acc, 4),
            "mean_labelwise_kappa": round(kappa,    4),
        },
        "mapping_results": {
            eid: {
                "cluster_id":        mr.cluster_id,
                "cluster_label":     mr.cluster_label,
                "top1_gold":         mr.top1_gold,
                "top1_similarity":   mr.top1_similarity,
                "majority_gold":     mr.majority_gold,
                "is_top1_correct":   mr.is_top1_correct,
                "ranked_candidates": [[lbl, round(sim, 6)]
                                      for lbl, sim in mr.ranked_candidates],
            }
            for eid, mr in mapping_results.items()
        },
        "centroid_similarity_matrix": {
            "cluster_ids":      entity_ids,
            "cluster_labels":   [entity_labels[eid] for eid in entity_ids],
            "canonical_labels": canonical,
            "values":           sim_matrix.tolist(),
        },
        "label_similarity_matrix": {
            "cluster_ids":      entity_ids,
            "cluster_labels":   [entity_labels[eid] for eid in entity_ids],
            "canonical_labels": canonical,
            "values":           label_sim_matrix.tolist(),
        },
        "matching_metrics": {
            "top1_accuracy_centroid":     round(top1_acc,        4),
            "top1_accuracy_label_string": round(label_top1_acc,  4),
            "soft_accuracy":              round(soft_acc,         4),
            "pairwise_auc":      round(pairwise_auc, 4) if pairwise_auc is not None else None,
            "mean_rougeL":       round(mean_rougeL,  4),
            "mean_bertscore_f1": round(mean_bs_f1,   4),
        },
        "global_metrics": {
            "sentence_accuracy":    round(sent_acc, 4),
            "mean_labelwise_kappa": round(kappa,    4),
        },
        "sentence_results": sentence_rows,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-level Theme Recovery Evaluation"
    )
    parser.add_argument("--gold-csv",        default=DEFAULT_GOLD_CSV,
                        help="Gold CSV (or adapted CSV for esconv). Use --indicator-col / --dimension-col to remap columns.")
    parser.add_argument("--clustered",       default=DEFAULT_CLUSTERED)
    parser.add_argument("--codebook",        default=DEFAULT_CODEBOOK)
    parser.add_argument("--meta-themes",     default=DEFAULT_META_THEMES)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--output",          default="theme_recovery_results.json")
    parser.add_argument("--indicator-col",       default="indicator",
                        help="CSV column for fine-grained gold labels (Level 1)")
    parser.add_argument("--extra-indicator-col", default="",
                        help="Comma-separated extra CSV columns merged into the Level-1 gold set. "
                             "ESconv: '--extra-indicator-col seeker_problem,supporter_strategy' merges "
                             "emotion type, problem type, and supporter strategy into one fine-grained level.")
    parser.add_argument("--dimension-col",       default="dimension",
                        help="CSV column for coarse-grained gold labels (Level 2). "
                             "Empty string to use --dim-map or skip Level 2.")
    parser.add_argument("--text-col",            default="text",
                        help="CSV column for the text/sentence content (default: 'text'). "
                             "Set to 'ad_creative_body' for climate data.")
    parser.add_argument("--dim-map",             default="",
                        help="JSON string or @file.json mapping Level-1 labels to Level-2 labels. "
                             "Covers ALL fine-grained labels from both seeker and supporter in one map. "
                             "Climate: '{\"CA\":\"Community & Resilience\",...}'. "
                             "ESconv: '{\"depression\":\"emotion\",\"job crisis\":\"problem\","
                             "\"Affirmation and Reassurance\":\"Comforting\",\"Information\":\"Action\",...}'")
    args = parser.parse_args()

    def _parse_map(raw: str) -> Dict[str, str]:
        raw = raw.strip()
        if not raw:
            return {}
        return json.load(open(raw[1:], encoding="utf-8")) if raw.startswith("@") else json.loads(raw)

    dim_map:    Dict[str, str] = _parse_map(args.dim_map)
    extra_cols: List[str]      = [c.strip() for c in args.extra_indicator_col.split(",") if c.strip()]
    run_level2  = bool(args.dimension_col) or bool(dim_map)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[0] Loading embedding model  ({args.embedding_model})")
    emb_model = SentenceTransformer(args.embedding_model, device="cpu")

    # ── Load shared data ──────────────────────────────────────────────────────
    print(f"\n[1] Loading clustered codes  ({args.clustered})")
    codes_per_review, code_to_cluster = load_clustered(args.clustered)

    print(f"\n[2] Pre-embedding all sentence texts (shared by both levels)")
    _, _, id_to_text = load_gold_level(args.gold_csv, args.indicator_col, args.text_col, extra_cols)
    unique_texts = sorted({t for t in id_to_text.values() if t})
    text2idx     = {t: i for i, t in enumerate(unique_texts)}
    print(f"    Embedding {len(unique_texts)} texts ...")
    # Encode all texts once; reused by all four evaluation levels
    sent_embs = emb_model.encode(unique_texts, normalize_embeddings=True,
                                 show_progress_bar=True)

    # ── Build cluster / meta_theme coverage (shared across all levels) ─────────
    cluster_labels  = load_cluster_labels(args.codebook)
    cluster_to_sids = build_cluster_to_sids(codes_per_review, code_to_cluster)

    mt_labels, mt_to_clusters = load_meta_themes(args.meta_themes)
    mt_to_sids = build_meta_theme_to_sids(mt_to_clusters, cluster_to_sids)
    cluster_to_meta: Dict[str, str] = {}
    for mt_id, cluster_ids in mt_to_clusters.items():
        for cid in cluster_ids:
            cluster_to_meta[cid] = mt_id

    # ═══════════════════════════════════════════════════════════════════════════
    # LEVEL 1 (fine): cluster ↔ indicator
    # --extra-indicator-col merges additional columns into the gold set so that
    # seeker emotion, seeker problem, and supporter strategy all count as the
    # same fine-grained level (ESconv) without splitting into separate passes.
    # ═══════════════════════════════════════════════════════════════════════════
    ind_desc = args.indicator_col + (f" + {args.extra_indicator_col}" if extra_cols else "")
    print(f"\n{'='*70}")
    print(f"LEVEL 1 (fine): cluster  ↔  {ind_desc} (gold)")
    print(f"{'='*70}")

    gold_sets_ind, canonical_ind, _ = load_gold_level(
        args.gold_csv, args.indicator_col, args.text_col, extra_cols
    )
    print(f"  {len(gold_sets_ind)} rows  |  {len(canonical_ind)} gold labels  |  {len(cluster_labels)} clusters")
    for t in canonical_ind:
        print(f"    · {t}")

    level1 = run_level(
        level_name="cluster_indicator",
        entity_labels=cluster_labels,
        entity_to_sids=cluster_to_sids,
        gold_sets=gold_sets_ind,
        canonical=canonical_ind,
        id_to_text=id_to_text,
        emb_model=emb_model,
        sent_embs=sent_embs,
        unique_texts=unique_texts,
        text2idx=text2idx,
        pred_sets_fn=lambda mr: build_pred_sets_cluster(
            codes_per_review, code_to_cluster, mr
        ),
        embedding_model_path=args.embedding_model,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # LEVEL 2 (coarse): meta_theme ↔ dimension
    # --dim-map covers ALL fine-grained labels in one mapping: climate codes
    # (CA → "Community & Resilience") and ESconv labels (depression → "emotion",
    # job crisis → "problem", Affirmation → "Comforting") alike.
    # ═══════════════════════════════════════════════════════════════════════════
    if run_level2:
        if dim_map:
            gold_sets_dim, canonical_dim = derive_dim_gold(gold_sets_ind, dim_map)
            level2_label = "derived from --dim-map"
        else:
            gold_sets_dim, canonical_dim, _ = load_gold_level(
                args.gold_csv, args.dimension_col, args.text_col
            )
            level2_label = args.dimension_col

        print(f"\n{'='*70}")
        print(f"LEVEL 2 (coarse): meta_theme  ↔  {level2_label} (gold)")
        print(f"{'='*70}")
        print(f"  {len(gold_sets_dim)} rows  |  {len(canonical_dim)} gold dimensions  |  {len(mt_labels)} meta_themes")
        for t in canonical_dim:
            print(f"    · {t}")

        level2 = run_level(
            level_name="meta_theme_dimension",
            entity_labels=mt_labels,
            entity_to_sids=mt_to_sids,
            gold_sets=gold_sets_dim,
            canonical=canonical_dim,
            id_to_text=id_to_text,
            emb_model=emb_model,
            sent_embs=sent_embs,
            unique_texts=unique_texts,
            text2idx=text2idx,
            pred_sets_fn=lambda mr: build_pred_sets_meta(
                codes_per_review, code_to_cluster, cluster_to_meta, mr
            ),
            embedding_model_path=args.embedding_model,
        )
    else:
        print(f"\n[LEVEL 2 skipped — no --dimension-col or --dim-map]")
        level2 = None

    # ── Save combined results ──────────────────────────────────────────────────
    output = {
        "level1_cluster_indicator":    level1,
        "level2_meta_theme_dimension": level2,
        # Flat aliases so existing generate_report.py works for Level 1
        "config":                     level1["config"],
        "summary":                    level1["summary"],
        "mapping_results":            level1["mapping_results"],
        "centroid_similarity_matrix": level1["centroid_similarity_matrix"],
        "label_similarity_matrix":    level1["label_similarity_matrix"],
        "matching_metrics":           level1["matching_metrics"],
        "global_metrics":             level1["global_metrics"],
        "sentence_results":           level1["sentence_results"],
    }

    out_path = os.path.join(HERE, args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
