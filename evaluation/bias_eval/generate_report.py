#!/usr/bin/env python3
"""
generate_report.py
==================
Reads theme_recovery_results.json (two-level output) and produces a
self-contained HTML report with metrics, mapping tables, confusion matrices,
and similarity matrices for BOTH evaluation levels.

Usage:
    python generate_report.py [--results results.json] [--output report.html]
"""

import argparse
import base64
import io
import subprocess
import sys
import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Short label maps ───────────────────────────────────────────────────────────
SHORTEN_INDICATOR = {
    "brooding over schoolwork during free time":           "brooding",
    "feeling inadequate in schoolwork":                    "feeling inadequate",
    "loss of interest":                                    "loss of interest",
    "low motivation / thoughts of giving up":              "low motivation",
    "lower expectations for one's schoolwork than before": "lower expectations",
    "overwhelmed by schoolwork":                           "overwhelmed",
    "poor sleep due to schoolwork":                        "poor sleep",
    "questioning the meaning of schoolwork":               "questioning meaning",
    "school pressure harming close relationships":         "school pressure",
}

SHORTEN_DIMENSION = {
    "Cynicism toward the meaning of school": "Cynicism",
    "Exhaustion at school":                  "Exhaustion",
    "Sense of inadequacy at school":         "Inadequacy",
}


def short_ind(label):
    return SHORTEN_INDICATOR.get(label, label)


def short_dim(label):
    return SHORTEN_DIMENSION.get(label, label)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_confusion(level_data: dict, shorten_fn) -> str:
    canonical   = level_data["config"]["canonical_themes"]
    short_canon = [shorten_fn(l) for l in canonical]
    idx         = {l: i for i, l in enumerate(canonical)}

    conf = np.zeros((len(canonical), len(canonical)), dtype=int)
    for mr in level_data["mapping_results"].values():
        mg = mr["majority_gold"]
        t1 = mr["top1_gold"]
        if mg is None:
            continue
        conf[idx[mg], idx[t1]] += 1

    fig, ax = plt.subplots(figsize=(max(6, len(canonical) * 1.2),
                                    max(5, len(canonical) * 1.0)))
    sns.heatmap(
        conf, ax=ax,
        xticklabels=short_canon, yticklabels=short_canon,
        cmap="Blues", annot=True, fmt="d",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted (top1_gold)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True (majority_gold)",  fontsize=12, fontweight="bold", labelpad=10)
    level_name = level_data["config"]["level"].replace("_", " → ")
    ax.set_title(f"Confusion matrix  [{level_name}]",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=9, rotation=35)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    return fig_to_b64(fig)


def plot_sim_matrix(level_data: dict, shorten_fn, row_truncate=35) -> str:
    sm         = level_data["centroid_similarity_matrix"]
    values     = np.array(sm["values"])
    row_labels = [lbl[:row_truncate] for lbl in sm["cluster_labels"]]
    col_labels = [shorten_fn(lbl) for lbl in sm["canonical_labels"]]

    fig_h = max(6, len(row_labels) * 0.5)
    fig_w = max(8, len(col_labels) * 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        values, ax=ax,
        xticklabels=col_labels, yticklabels=row_labels,
        cmap="YlOrRd",
        vmin=values.min(), vmax=values.max(),
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Cosine similarity"},
    )
    level_name = level_data["config"]["level"].replace("_", " → ")
    ax.set_xlabel("Gold label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Synthetic entity", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"Centroid similarity matrix  [{level_name}]",
                 fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(axis="x", labelsize=9, rotation=30)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig_to_b64(fig)


def plot_per_label(level_data: dict, shorten_fn) -> str:
    canonical = level_data["config"]["canonical_themes"]
    gold_sets = {r["sentence_id"]: set(r["gold_set"]) for r in level_data["sentence_results"]}
    pred_sets = {r["sentence_id"]: set(r["pred_set"]) for r in level_data["sentence_results"]}

    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)

    for sid, gold in gold_sets.items():
        pred = pred_sets.get(sid, set())
        for lbl in canonical:
            g = lbl in gold
            p = lbl in pred
            if g and p:       label_tp[lbl] += 1
            elif p and not g: label_fp[lbl] += 1
            elif g and not p: label_fn[lbl] += 1

    precs, recs, f1s, labels = [], [], [], []
    for lbl in canonical:
        tp, fp, fn = label_tp[lbl], label_fp[lbl], label_fn[lbl]
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        precs.append(pr); recs.append(rc); f1s.append(f1)
        labels.append(shorten_fn(lbl))

    x   = np.arange(len(labels))
    w   = 0.25
    fig_w = max(8, len(labels) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(x - w, precs, w, label="Precision", color="#4C72B0")
    ax.bar(x,     recs,  w, label="Recall",    color="#55A868")
    ax.bar(x + w, f1s,   w, label="F1",        color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    level_name = level_data["config"]["level"].replace("_", " → ")
    ax.set_title(f"Per-label Precision / Recall / F1  [{level_name}]", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig_to_b64(fig)


# ── HTML builder ───────────────────────────────────────────────────────────────

CSS = """
body { font-family: "Segoe UI", Arial, sans-serif; margin: 40px auto; max-width: 1200px;
       color: #222; background: #fafafa; }
h1   { font-size: 1.8em; border-bottom: 3px solid #3a6ea5; padding-bottom: 8px; color: #1a3a5c; }
h2   { font-size: 1.35em; margin-top: 48px; color: #2c5282;
       border-left: 4px solid #3a6ea5; padding-left: 10px; }
h3   { font-size: 1.1em; margin-top: 28px; color: #2d3748; }
.level-banner { background: #ebf4ff; border: 1px solid #bee3f8; border-radius: 6px;
                padding: 10px 16px; margin: 24px 0 8px; font-weight: bold;
                font-size: 1.05em; color: #2b6cb0; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.9em; }
th    { background: #3a6ea5; color: white; padding: 8px 12px; text-align: left; }
td    { padding: 7px 12px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) { background: #f0f4f9; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
               gap: 16px; margin: 20px 0; }
.metric-card { background: white; border: 1px solid #cce; border-radius: 8px;
               padding: 16px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.metric-card .val  { font-size: 2em; font-weight: bold; color: #2b6cb0; }
.metric-card .name { font-size: 0.85em; color: #666; margin-top: 4px; }
.correct   { color: #276; font-weight: bold; }
.incorrect { color: #c22; }
img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 12px 0; }
.footer { margin-top: 60px; font-size: 0.8em; color: #999; text-align: center; }
hr.level-sep { border: none; border-top: 2px solid #bee3f8; margin: 48px 0; }
"""


def metric_card(name, value):
    return f'<div class="metric-card"><div class="val">{value}</div><div class="name">{name}</div></div>'


def build_level_section(level_data: dict, level_num: int, shorten_fn,
                        conf_b64: str, sim_b64: str, label_b64: str) -> str:
    s   = level_data["summary"]
    cfg = level_data["config"]
    n_entities  = len(level_data["mapping_results"])
    n_sentences = len(level_data["sentence_results"])
    n_canonical = len(cfg["canonical_themes"])
    level_label = cfg["level"].replace("_", " → ")

    entity_word  = "Meta-theme" if "meta_theme" in cfg["level"] else "Cluster"
    gold_word    = "dimension"  if "meta_theme" in cfg["level"] else "indicator"

    cards = "".join([
        metric_card("Top-1 Acc (Centroid)",   f'{s["top1_accuracy_centroid"]:.4f}'),
        metric_card("Top-1 Acc (Label)",      f'{s["top1_accuracy_label_string"]:.4f}'),
        metric_card("Soft Accuracy",
                    f'{s["soft_accuracy"]:.4f}' if s.get("soft_accuracy") is not None else "—"),
        metric_card("Pairwise AUC",
                    f'{s["pairwise_auc"]:.4f}' if s.get("pairwise_auc") is not None else "N/A"),
        metric_card("Mean ROUGE-L",       f'{s["mean_rougeL"]:.4f}'),
        metric_card("Mean BERTScore F1",  f'{s["mean_bertscore_f1"]:.4f}'),
        metric_card("Sentence Accuracy",  f'{s["sentence_accuracy"]:.4f}'),
        metric_card("Labelwise Kappa",    f'{s["mean_labelwise_kappa"]:.4f}'),
    ])

    # Entity mapping table
    rows = []
    for mr in sorted(level_data["mapping_results"].values(),
                     key=lambda x: int(x["cluster_id"])):
        correct = mr["is_top1_correct"]
        if correct is True:
            flag = '<span class="correct">✓</span>'
        elif correct is False:
            flag = '<span class="incorrect">✗</span>'
        else:
            flag = "—"
        rows.append(
            f"<tr>"
            f"<td>{mr['cluster_id']}</td>"
            f"<td>{mr['cluster_label']}</td>"
            f"<td>{mr['top1_gold']}</td>"
            f"<td>{mr['majority_gold'] or '—'}</td>"
            f"<td>{mr['top1_similarity']:.4f}</td>"
            f"<td>{flag}</td>"
            f"</tr>"
        )
    entity_table = "\n".join(rows)

    return f"""
<div class="level-banner">Level {level_num}: {level_label.upper()}
  &nbsp;|&nbsp; {n_sentences} sentences &nbsp;|&nbsp;
  {n_entities} {entity_word.lower()}s &nbsp;|&nbsp;
  {n_canonical} gold {gold_word}s</div>

<h3>Summary Metrics</h3>
<div class="metric-grid">{cards}</div>

<h3>{entity_word}-level Mapping Details</h3>
<table>
  <thead>
    <tr><th>#</th><th>{entity_word} label</th><th>Predicted (top1_gold)</th>
        <th>True (majority_gold)</th><th>Similarity</th><th>Correct?</th></tr>
  </thead>
  <tbody>{entity_table}</tbody>
</table>

<h3>Confusion Matrix</h3>
<img src="data:image/png;base64,{conf_b64}" alt="Confusion matrix L{level_num}">

<h3>Per-label Precision / Recall / F1 (sentence-level)</h3>
<img src="data:image/png;base64,{label_b64}" alt="Per-label chart L{level_num}">

<h3>Centroid Similarity Matrix</h3>
<img src="data:image/png;base64,{sim_b64}" alt="Similarity matrix L{level_num}">
"""


def build_html(d: dict) -> str:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    emb_name = os.path.basename(d["level1_cluster_indicator"]["config"]["embedding_model"])

    l1 = d["level1_cluster_indicator"]
    l2 = d["level2_meta_theme_dimension"]

    print("  [L1] Generating confusion matrix ...")
    l1_conf  = plot_confusion(l1, short_ind)
    print("  [L1] Generating similarity matrix ...")
    l1_sim   = plot_sim_matrix(l1, short_ind, row_truncate=40)
    print("  [L1] Generating per-label chart ...")
    l1_label = plot_per_label(l1, short_ind)

    print("  [L2] Generating confusion matrix ...")
    l2_conf  = plot_confusion(l2, short_dim)
    print("  [L2] Generating similarity matrix ...")
    l2_sim   = plot_sim_matrix(l2, short_dim, row_truncate=50)
    print("  [L2] Generating per-label chart ...")
    l2_label = plot_per_label(l2, short_dim)

    section_l1 = build_level_section(l1, 1, short_ind, l1_conf, l1_sim, l1_label)
    section_l2 = build_level_section(l2, 2, short_dim, l2_conf, l2_sim, l2_label)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Theme Recovery Evaluation Report (Two-Level)</title>
<style>{CSS}</style>
</head>
<body>

<h1>Theme Recovery Evaluation Report</h1>
<p><strong>Generated:</strong> {run_time} &nbsp;|&nbsp;
   <strong>Embedding model:</strong> {emb_name}</p>

<h2>Level 1 — cluster ↔ indicator (fine-grained)</h2>
{section_l1}

<hr class="level-sep">

<h2>Level 2 — meta_theme ↔ dimension (coarse)</h2>
{section_l2}

<div class="footer">Auto-generated by generate_report.py</div>
</body>
</html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(HERE, "theme_recovery_results.json"))
    parser.add_argument("--output",  default=os.path.join(HERE, "theme_recovery_report.html"))
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation and use existing results JSON directly")
    args = parser.parse_args()

    if not args.skip_eval:
        eval_script = os.path.join(HERE, "evaluate_theme_recovery.py")
        print("=== Running evaluation ===")
        subprocess.run(
            [sys.executable, eval_script, "--output", args.results],
            check=True,
        )
        print("=== Evaluation done ===\n")

    print(f"Loading {args.results} ...")
    d = json.load(open(args.results, encoding="utf-8"))

    # Handle old single-level results gracefully
    if "level1_cluster_indicator" not in d:
        print("WARNING: results file is old single-level format; wrapping as level1 only.")
        d = {
            "level1_cluster_indicator":    d,
            "level2_meta_theme_dimension": d,
        }

    print("Building HTML report ...")
    html = build_html(d)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
