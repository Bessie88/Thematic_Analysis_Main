"""LangChain tools for the grounded-theory pipeline."""

import json
import os
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .embeddings import encode_texts
from .hierarchy_refine import maybe_refine_hierarchy
from .inference_config import llm_model_name, openai_base
from .llm_clustering import axial_llm_cluster, use_llm_clustering
from .paths import (
    CLUSTERED_CODES_PATH,
    CODEBOOK_CONFIDENCE_PATH,
    CODEBOOK_PATH,
    CODEBOOK_PROVENANCE_PATH,
    GLOBAL_GRAPH_PATH,
    HIERARCHY_PATH,
    META_THEMES_PATH,
    display_path,
    ensure_output_dirs,
)
from .pipeline_helpers import (
    EMBED_DRAIN_THRESHOLD,
    REFINE_TOP_K_OTHER_CLUSTERS,
    assign_codes_to_subthemes_prompt,
    axial_embed_and_cluster,
    build_sub_theme_node,
    drain_ungrouped_to_subthemes,
    hierarchy_assign_batch,
    hierarchy_embed_drain_enabled,
    meta_theme_bounds,
    normalize_meta_theme_count,
    prune_hierarchy_to_valid_clusters,
    refine_llm_max_codes,
    save_hierarchy,
)
from .prompts import (
    high_level_code_generation_prompt,
    high_level_synthesis_prompt,
    intra_cluster_subtheme_prompt,
    meta_theme_grouping_prompt,
    open_coding_prompt,
    refine_cluster_assignments_prompt,
    validate_open_codes_prompt,
)
from .skills import llm_invoke_with_skill
from .utils import clean_and_parse_json, log_step, remove_think_tags

COMPLETION_TOKENS = 4096

llm = ChatOpenAI(
    model=llm_model_name(),
    openai_api_key="EMPTY",
    openai_api_base=openai_base(),
    temperature=0,
    max_tokens=COMPLETION_TOKENS,
    model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
)

# HICode-style axial clustering (temperature=0, up to 8192 completion tokens).
# Each batch call passes a dynamic max_tokens so input + completion fit the SGLang
# context window (see _llm_cluster_completion_budget in llm_clustering.py).
cluster_llm = ChatOpenAI(
    model="llm",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    temperature=0,
    max_tokens=8192,
    model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
)


@tool
def open_coding(text: str, research_question: str, validator_feedback: Optional[str] = None) -> str:
    """
    Open coding for one text unit: induce 0–3 codes aligned with the research question.
    Use Applicability NONE when irrelevant; if validator_feedback is set, revise the prior attempt.
    """
    prompt = open_coding_prompt(research_question, text, validator_feedback=validator_feedback)
    return llm_invoke_with_skill(llm, "open_coding", prompt)


@tool
def validate_open_codes(text: str, generated_codes: str, research_question: str) -> str:
    """Validate one open-coding result; respond PASS or FAIL with actionable feedback."""
    prompt = validate_open_codes_prompt(research_question, text, generated_codes)
    return llm_invoke_with_skill(llm, "validate_open_codes", prompt)


@tool
def axial_coding(open_codes: str, research_question: str = "") -> str:
    """Axial step: JSON array of code strings → cluster; write gt_clustered_codes.json."""
    try:
        all_codes = json.loads(open_codes)
    except json.JSONDecodeError:
        return "axial_coding expects a JSON array of code strings."
    if not isinstance(all_codes, list) or not all(isinstance(x, str) for x in all_codes):
        return "axial_coding expects a JSON array of code strings."
    out_dir = str(CLUSTERED_CODES_PATH.parent)
    if use_llm_clustering():
        return axial_llm_cluster(all_codes, research_question, cluster_llm, out_dir=out_dir)
    return axial_embed_and_cluster(all_codes, out_dir=out_dir)


@tool
def high_level_code_generation(
    cluster_file: str = str(CLUSTERED_CODES_PATH), research_question: str = ""
) -> str:
    """
    One LLM call per cluster: label, confidence (1–5), rationale.
    Writes codebook.json and codebook_confidence.json; returns codebook JSON.
    """
    if not os.path.isfile(cluster_file):
        return json.dumps({"error": f"Missing {cluster_file}; run axial step first."})
    with open(cluster_file, encoding="utf-8") as f:
        data = json.load(f)
    cluster_to_codes = data.get("cluster_to_codes", {})
    if not cluster_to_codes:
        return json.dumps({"error": "No cluster_to_codes in file."})
    out_dir = os.path.dirname(cluster_file) or "."
    out_path = os.path.join(out_dir, "codebook.json")
    confidence_path = os.path.join(out_dir, "codebook_confidence.json")

    theme_names = data.get("cluster_theme_names")
    if isinstance(theme_names, dict) and theme_names:
        codebook = {}
        codebook_confidence: Dict[str, Dict[str, Any]] = {}
        for cid in sorted(cluster_to_codes.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
            label = str(theme_names.get(cid, theme_names.get(str(cid), ""))).strip()
            if not label:
                label = f"Cluster {cid}"
            codebook[str(cid)] = label
            codebook_confidence[str(cid)] = {
                "label": label,
                "confidence": 4,
                "rationale": "from LLM axial clustering",
            }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"codebook": codebook, "cluster_to_codes": cluster_to_codes}, f, indent=2)
        with open(confidence_path, "w", encoding="utf-8") as f:
            json.dump(codebook_confidence, f, indent=2)
        log_step(
            "HIGH_LEVEL_SKIPPED",
            f"Using cluster_theme_names from LLM clustering ({len(codebook)} clusters).",
        )
        return json.dumps(codebook)
    codebook = {}
    codebook_confidence: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            codebook = existing.get("codebook", {})
        except (json.JSONDecodeError, KeyError):
            pass
    if os.path.isfile(confidence_path):
        try:
            with open(confidence_path, encoding="utf-8") as f:
                codebook_confidence = json.load(f)
        except (json.JSONDecodeError, TypeError):
            codebook_confidence = {}

    def _label_one(cid, codes_list):
        """Call LLM for a single cluster; returns (label, confidence, rationale)."""
        if not codes_list:
            return f"Cluster {cid}", 1, ""
        bulleted = "\n".join(f"- {c}" for c in codes_list[:30])
        if len(codes_list) > 30:
            bulleted += f"\n- ... and {len(codes_list) - 30} more"
        prompt = high_level_code_generation_prompt(bulleted, research_question)
        try:
            raw = llm_invoke_with_skill(llm, "high_level_code_generation", prompt, cluster_id=cid)
            parsed = clean_and_parse_json(remove_think_tags(raw))
            label = (parsed.get("label") or "").strip().strip("\"'")
            if not label or len(label) > 80:
                label = f"Cluster {cid}"
            confidence = parsed.get("confidence", 1)
            if not isinstance(confidence, int):
                try:
                    confidence = int(float(confidence))
                except (TypeError, ValueError):
                    confidence = 1
            confidence = max(1, min(5, confidence))
            rationale = (parsed.get("rationale") or "").strip()
            return label, confidence, rationale
        except Exception as e:
            log_step("HIGH_LEVEL_LLM_ERROR", f"Cluster {cid}: {e}")
            return f"Cluster {cid}", 1, ""

    # Strategy for high-level code generation:
    # - default: first 30 codes in the cluster
    # - nsampling: draw n random samples, get a candidate label per sample, synthesize.
    strategy = os.environ.get("GT_HIGH_LEVEL_STRATEGY", "nsampling")
    n_samples = int(os.environ.get("GT_HL_N_SAMPLES", "5"))
    sample_size = int(os.environ.get("GT_HL_SAMPLE_SIZE", "15"))

    def _label_one_nsampling(cid, codes_list):
        """N-sampling strategy: draw n random samples, get a candidate label per sample, synthesize.

        Returns a 4-tuple (label, confidence, rationale, candidate_labels) so callers can store
        the intermediate candidates for post-hoc analysis of inter-sample agreement.
        """
        if not codes_list:
            return f"Cluster {cid}", 1, "", []
        if len(codes_list) <= sample_size:
            label, confidence, rationale = _label_one(cid, codes_list)
            return label, confidence, rationale, []
        candidate_labels = []
        for _ in range(n_samples):
            sample = random.sample(codes_list, min(sample_size, len(codes_list)))
            bulleted = "\n".join(f"- {c}" for c in sample)
            prompt = high_level_code_generation_prompt(bulleted, research_question)
            try:
                raw = llm_invoke_with_skill(
                    llm, "high_level_code_generation", prompt, cluster_id=cid
                )
                parsed = clean_and_parse_json(remove_think_tags(raw))
                lb = (parsed.get("label") or "").strip().strip("\"'")
                if lb and len(lb) <= 80:
                    candidate_labels.append(lb)
            except Exception as e:
                log_step("HIGH_LEVEL_LLM_ERROR", f"Cluster {cid} sample: {e}")
        if not candidate_labels:
            return f"Cluster {cid}", 1, "", []
        prompt = high_level_synthesis_prompt(candidate_labels, research_question)
        try:
            raw = llm_invoke_with_skill(llm, "high_level_synthesis", prompt, cluster_id=cid)
            parsed = clean_and_parse_json(remove_think_tags(raw))
            label = (parsed.get("label") or "").strip().strip("\"'")
            if not label or len(label) > 80:
                label = candidate_labels[0]
            confidence = parsed.get("confidence", 1)
            if not isinstance(confidence, int):
                try:
                    confidence = int(float(confidence))
                except (TypeError, ValueError):
                    confidence = 1
            confidence = max(1, min(5, confidence))
            rationale = (parsed.get("rationale") or "").strip()
            return label, confidence, rationale, candidate_labels
        except Exception as e:
            log_step("HIGH_LEVEL_LLM_ERROR", f"Cluster {cid} synthesis: {e}")
            return candidate_labels[0], 1, "", candidate_labels

    # Fix up any clusters already in codebook but missing confidence entry (no LLM needed)
    for cid in list(codebook.keys()):
        if cid not in codebook_confidence:
            codebook_confidence[cid] = {"label": codebook[cid], "confidence": 1, "rationale": ""}

    pending = {
        cid: (codes if isinstance(codes, list) else [])
        for cid, codes in sorted(cluster_to_codes.items(), key=lambda x: int(x[0]))
        if cid not in codebook
    }

    _hl_lock = threading.Lock()
    workers = int(os.environ.get("GT_HIGH_LEVEL_WORKERS", "8"))
    _use_nsampling = strategy == "nsampling"
    _fn = _label_one_nsampling if _use_nsampling else _label_one
    log_step(
        "HIGH_LEVEL_STRATEGY",
        f"strategy={strategy!r} | n_samples={n_samples} | sample_size={sample_size} | workers={workers}"
        if _use_nsampling
        else f"strategy={strategy!r} (first-30 default)",
    )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fn, cid, codes): cid for cid, codes in pending.items()}
        for fut in as_completed(futures):
            cid = futures[fut]
            result = fut.result()
            label, confidence, rationale = result[0], result[1], result[2]
            candidates = result[3] if len(result) > 3 else None
            with _hl_lock:
                codebook[cid] = label
                confidence_entry: Dict[str, Any] = {
                    "label": label,
                    "confidence": confidence,
                    "rationale": rationale,
                }
                if candidates is not None:
                    confidence_entry["candidate_labels"] = candidates
                codebook_confidence[cid] = confidence_entry
                if confidence <= 2:
                    log_step(
                        "LOW_CONFIDENCE_CLUSTER",
                        f"Cluster {cid}: confidence={confidence}, label={label}",
                    )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"codebook": codebook, "cluster_to_codes": cluster_to_codes}, f, indent=2
                    )
                with open(confidence_path, "w", encoding="utf-8") as f:
                    json.dump(codebook_confidence, f, indent=2)

    return json.dumps(codebook)


@tool
def refine_cluster_assignments(codebook_path: str, cluster_file: str) -> str:
    """
    LLM-suggested MOVEs between clusters; rekeys clusters and syncs codebook / confidence / hierarchy keys.
    """
    if not os.path.isfile(codebook_path):
        return f"Error: codebook not found at {codebook_path}"
    if not os.path.isfile(cluster_file):
        return f"Error: cluster file not found at {cluster_file}"
    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)
    codebook = cb_data.get("codebook", {})
    with open(cluster_file, encoding="utf-8") as f:
        data = json.load(f)
    cluster_to_codes = data.get("cluster_to_codes", {})
    all_codes = data.get("all_codes", [])
    codes_per_review = data.get("codes_per_review", [])

    label_to_cids: Dict[str, List[str]] = defaultdict(list)
    for cid, label in codebook.items():
        label_to_cids[label].append(cid)

    moves_applied: List[tuple] = []

    sorted_oids = sorted(cluster_to_codes.keys(), key=lambda x: int(x))
    if not sorted_oids:
        return "Error: no clusters in cluster file"

    label_for_oid = {oid: codebook.get(oid, f"Cluster {oid}") for oid in sorted_oids}
    label_texts = [label_for_oid[oid] for oid in sorted_oids]

    try:
        emb = encode_texts(label_texts, normalize=True, show_progress=False)
    except Exception as e:
        return f"Error embedding cluster labels for refine: {e}"
    oid_to_pos = {oid: i for i, oid in enumerate(sorted_oids)}

    skip_refine: set = set()
    if os.path.isfile(str(CODEBOOK_PROVENANCE_PATH)):
        try:
            with open(CODEBOOK_PROVENANCE_PATH, encoding="utf-8") as f:
                prov = json.load(f)
            skip_refine = set(str(x) for x in prov.get("skip_refine_cluster_ids") or [])
        except (json.JSONDecodeError, OSError, TypeError):
            skip_refine = set()
    if not skip_refine and os.path.isfile(str(CODEBOOK_CONFIDENCE_PATH)):
        try:
            with open(CODEBOOK_CONFIDENCE_PATH, encoding="utf-8") as f:
                conf = json.load(f)
            for cid, entry in conf.items():
                if isinstance(entry, dict) and entry.get("needs_more_evidence"):
                    skip_refine.add(str(cid))
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    for cid in sorted_oids:
        if str(cid) in skip_refine:
            log_step("REFINE_SKIP_HUMAN_FLAG", f"Cluster {cid}: needs_more_evidence or human skip")
            continue
        codes = cluster_to_codes.get(cid, [])
        if not codes:
            continue
        label = codebook.get(cid, f"Cluster {cid}")
        pos = oid_to_pos[cid]
        sims = emb @ emb[pos]
        sims = sims.copy()
        sims[pos] = -np.inf

        n_others = len(sorted_oids) - 1
        k_take = min(REFINE_TOP_K_OTHER_CLUSTERS, n_others)
        if k_take <= 0:
            other_str = "(none)"
        else:
            part = np.argpartition(-sims, k_take - 1)[:k_take]
            top_positions = part[np.argsort(-sims[part])]
            other_labels = [label_texts[j] for j in top_positions]
            other_str = ", ".join(other_labels)

        max_chunk = refine_llm_max_codes()
        chunk_codes_set = set(codes)
        for chunk_idx, chunk_start in enumerate(range(0, len(codes), max_chunk)):
            chunk = codes[chunk_start : chunk_start + max_chunk]
            bulleted = "\n".join(f"- {c}" for c in chunk)

            prompt = refine_cluster_assignments_prompt(label, bulleted, other_str)

            try:
                raw = remove_think_tags(
                    llm_invoke_with_skill(
                        llm,
                        "refine_cluster_assignments",
                        prompt,
                        cluster_id=cid,
                        chunk_index=chunk_idx,
                    )
                )
            except Exception as e:
                log_step("REFINE_LLM_ERROR", f"Cluster {cid} chunk {chunk_idx}: {e}")
                continue
            for line in raw.splitlines():
                line = line.strip()
                if line.upper() == "NONE" or not line:
                    continue
                match = re.search(
                    r'MOVE:\s*["\']([^"\']+)["\']\s*[→>]\s*["\']([^"\']+)["\']',
                    line,
                    re.IGNORECASE,
                )
                if not match:
                    continue
                code, target_label = match.group(1).strip(), match.group(2).strip()
                if code not in chunk_codes_set:
                    continue
                target_cids = label_to_cids.get(target_label, [])
                if not target_cids:
                    continue
                if len(target_cids) > 1:
                    log_step(
                        "REFINE_SKIP_AMBIGUOUS_LABEL",
                        f"MOVE skipped: label '{target_label}' maps to multiple clusters",
                    )
                    continue
                target_cid = target_cids[0]
                if target_cid == cid:
                    continue
                moves_applied.append((code, cid, target_cid))

    seen_codes: Dict[str, tuple] = {}
    deduped_moves: List[tuple] = []
    for move in moves_applied:
        code, from_cid, to_cid = move
        if code in seen_codes:
            log_step(
                "REFINE_CONFLICTING_MOVE",
                f"Code '{code}' has multiple move targets; keeping first.",
            )
        else:
            seen_codes[code] = move
            deduped_moves.append(move)
    moves_applied = deduped_moves

    code_to_cid: Dict[str, str] = {}
    for cid, codes in cluster_to_codes.items():
        for c in codes:
            code_to_cid[c] = cid
    for code, _from, to_cid in moves_applied:
        if code in code_to_cid:
            code_to_cid[code] = to_cid

    new_cluster_to_codes: Dict[str, List[str]] = defaultdict(list)
    for code, cid in code_to_cid.items():
        new_cluster_to_codes[cid].append(code)
    non_empty = {cid: codes for cid, codes in new_cluster_to_codes.items() if codes}
    cid_list = sorted(non_empty.keys(), key=int)
    new_cluster_to_codes = {str(i): non_empty[cid_list[i]] for i in range(len(cid_list))}
    k_new = len(new_cluster_to_codes)
    code_to_idx = {c: i for i, cids in enumerate(new_cluster_to_codes.values()) for c in cids}
    labels = [code_to_idx.get(c, 0) for c in all_codes]

    out = {
        "all_codes": all_codes,
        "labels": labels,
        "k": k_new,
        "cluster_to_codes": new_cluster_to_codes,
        "codes_per_review": codes_per_review,
    }
    os.makedirs(os.path.dirname(cluster_file) or ".", exist_ok=True)
    with open(cluster_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    new_codebook = {
        str(i): codebook.get(cid_list[i], f"Cluster {cid_list[i]}") for i in range(len(cid_list))
    }
    with open(codebook_path, "w", encoding="utf-8") as f:
        json.dump({"codebook": new_codebook, "cluster_to_codes": new_cluster_to_codes}, f, indent=2)
    confidence_path = os.path.join(
        os.path.dirname(codebook_path) or ".", "codebook_confidence.json"
    )
    codebook_confidence_rekeyed: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(confidence_path):
        try:
            with open(confidence_path, encoding="utf-8") as f:
                old_confidence = json.load(f)
            for i in range(len(cid_list)):
                old_cid = cid_list[i]
                entry = old_confidence.get(str(old_cid), old_confidence.get(old_cid))
                if isinstance(entry, dict):
                    codebook_confidence_rekeyed[str(i)] = {**entry, "label": new_codebook[str(i)]}
                else:
                    codebook_confidence_rekeyed[str(i)] = {
                        "label": new_codebook[str(i)],
                        "confidence": 1,
                        "rationale": "",
                    }
        except (json.JSONDecodeError, TypeError):
            codebook_confidence_rekeyed = {
                str(i): {"label": new_codebook[str(i)], "confidence": 1, "rationale": ""}
                for i in range(len(cid_list))
            }
    else:
        codebook_confidence_rekeyed = {
            str(i): {"label": new_codebook[str(i)], "confidence": 1, "rationale": ""}
            for i in range(len(cid_list))
        }
    with open(confidence_path, "w", encoding="utf-8") as f:
        json.dump(codebook_confidence_rekeyed, f, indent=2)

    hier_path = str(HIERARCHY_PATH)
    if os.path.isfile(hier_path):
        try:
            with open(hier_path, encoding="utf-8") as f:
                hier = json.load(f)
            if isinstance(hier, dict):
                valid_h = set(new_cluster_to_codes.keys())
                pruned, removed = prune_hierarchy_to_valid_clusters(hier, valid_h)
                if removed:
                    ensure_output_dirs()
                    with open(hier_path, "w", encoding="utf-8") as f:
                        json.dump(pruned, f, indent=2)
                    log_step(
                        "HIERARCHY_PRUNE_AFTER_REFINE",
                        f"Removed {len(removed)} stale cluster key(s) after rekey",
                    )
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    return f"Refined cluster assignments: {len(moves_applied)} codes moved across clusters."


@tool
def meta_theme_grouping(research_question: str = "") -> str:
    """Group cluster labels into a bounded set of meta-themes; writes gt_meta_themes.json."""
    codebook_path = str(CODEBOOK_PATH)
    if not os.path.isfile(codebook_path):
        return json.dumps({"error": "Missing codebook.json; run high-level step first."})
    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)
    codebook = cb_data.get("codebook", {})
    if not codebook:
        return json.dumps({"error": "No codebook in codebook.json."})

    labels_json = json.dumps(codebook, indent=2)
    all_cids = set(codebook.keys())
    n_cids = len(all_cids)
    lo, hi = meta_theme_bounds(n_cids)

    def _call_meta_llm(extra: str = "", phase: str = "initial") -> List[Dict[str, Any]]:
        base = meta_theme_grouping_prompt(labels_json, research_question)
        prompt = base + (extra or "")
        raw = llm_invoke_with_skill(llm, "meta_theme_grouping", prompt)
        parsed = clean_and_parse_json(remove_think_tags(raw))
        mt = parsed.get("meta_themes", [])
        if not isinstance(mt, list):
            return []
        return [m for m in mt if isinstance(m, dict)]

    try:
        meta_themes = _call_meta_llm()
    except Exception as e:
        log_step("META_THEME_LLM_ERROR", str(e))
        return json.dumps({"error": f"LLM call failed: {e}"})

    if not meta_themes:
        return json.dumps({"error": "LLM returned invalid meta_themes structure."})

    assigned_cids: set = set()
    for mt in meta_themes:
        for cid in mt.get("cluster_ids", []):
            assigned_cids.add(str(cid))

    missing = all_cids - assigned_cids
    if missing:
        largest = max(meta_themes, key=lambda m: len(m.get("cluster_ids", [])))
        for cid in missing:
            largest["cluster_ids"].append(str(cid))
        log_step(
            "META_THEME_FIXUP",
            f"Added {len(missing)} missing cluster(s) to group '{largest.get('name', '')}'",
        )

    if not (lo <= len(meta_themes) <= hi):
        reminder = (
            f"\n\nIMPORTANT: You must output between {lo} and {hi} meta_themes (inclusive), "
            f"each with a distinct name. Every cluster ID must appear exactly once."
        )
        try:
            meta_themes_retry = _call_meta_llm(reminder, phase="retry")
            if meta_themes_retry:
                meta_themes = meta_themes_retry
                assigned_cids = set()
                for mt in meta_themes:
                    for cid in mt.get("cluster_ids", []):
                        assigned_cids.add(str(cid))
                missing = all_cids - assigned_cids
                if missing:
                    largest = max(meta_themes, key=lambda m: len(m.get("cluster_ids", [])))
                    for cid in missing:
                        largest["cluster_ids"].append(str(cid))
                    log_step(
                        "META_THEME_FIXUP_RETRY",
                        f"Added {len(missing)} missing cluster(s) after retry",
                    )
        except Exception as e:
            log_step("META_THEME_RETRY_ERROR", str(e))

    if not (lo <= len(meta_themes) <= hi):
        before = len(meta_themes)
        meta_themes = normalize_meta_theme_count(meta_themes, n_cids)
        log_step(
            "META_THEME_REPAIR",
            f"Adjusted meta-theme count from {before} to {len(meta_themes)} (bounds [{lo},{hi}])",
        )

    ensure_output_dirs()
    with open(META_THEMES_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta_themes": meta_themes}, f, indent=2)

    summary = (
        f"Meta-themes: {len(meta_themes)} groups covering {len(all_cids)} clusters. "
        f"See {display_path(META_THEMES_PATH)}"
    )
    return summary


@tool
def hierarchy_construction(research_question: str = "") -> str:
    """Intra-cluster sub-themes (LLM + optional batched assign + embed drain); writes gt_hierarchy.json."""
    codebook_path = str(CODEBOOK_PATH)
    if not os.path.isfile(codebook_path):
        return json.dumps({"error": "Missing codebook.json; run high-level step first."})
    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)
    codebook = cb_data.get("codebook", {})
    cluster_to_codes = cb_data.get("cluster_to_codes", {})
    if not cluster_to_codes:
        return json.dumps({"error": "No cluster_to_codes in codebook.json."})

    existing_hierarchy: Dict[str, Any] = {}
    if os.path.isfile(str(HIERARCHY_PATH)):
        try:
            with open(HIERARCHY_PATH, encoding="utf-8") as f:
                existing_hierarchy = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_hierarchy = {}

    hierarchy: Dict[str, Any] = dict(existing_hierarchy)
    valid_ids = set(cluster_to_codes.keys())
    hierarchy, pruned_keys = prune_hierarchy_to_valid_clusters(hierarchy, valid_ids)
    if pruned_keys:
        log_step(
            "HIERARCHY_PRUNE",
            f"Removed {len(pruned_keys)} stale cluster key(s) not in codebook: "
            f"{pruned_keys[:20]}{'...' if len(pruned_keys) > 20 else ''}",
        )
        save_hierarchy(hierarchy)

    def _process_one(cid, codes, label):
        """Build hierarchy entry for a single cluster; all LLM calls are self-contained."""
        unique_codes = list(dict.fromkeys(codes))

        if len(unique_codes) <= 5:
            return cid, {"label": label, "sub_themes": [], "ungrouped_codes": unique_codes}

        BATCH_SIZE = 60
        codes_for_prompt = unique_codes[:BATCH_SIZE]
        prompt = intra_cluster_subtheme_prompt(
            label, "\n".join(f"- {c}" for c in codes_for_prompt), research_question
        )
        try:
            raw = llm_invoke_with_skill(
                llm, "hierarchy_construction", prompt, cluster_id=cid, phase="subtheme"
            )
            parsed = clean_and_parse_json(raw)
        except Exception as e:
            log_step("HIERARCHY_LLM_ERROR", f"Cluster {cid}: {e}")
            return cid, {"label": label, "sub_themes": [], "ungrouped_codes": unique_codes}

        assigned_codes: set = set()
        validated_sub_themes: List[Dict[str, Any]] = []
        for st in parsed.get("sub_themes", []):
            if not isinstance(st, dict):
                continue
            st_codes = [c for c in st.get("codes", []) if c in set(codes_for_prompt)]
            assigned_codes.update(st_codes)
            if st_codes:
                validated_sub_themes.append({"name": st.get("name", "Unnamed"), "codes": st_codes})

        validated_ungrouped = [
            c
            for c in parsed.get("ungrouped_codes", [])
            if c in set(codes_for_prompt) and c not in assigned_codes
        ]
        assigned_codes.update(validated_ungrouped)
        validated_ungrouped.extend(c for c in codes_for_prompt if c not in assigned_codes)

        if len(unique_codes) > BATCH_SIZE:
            remaining = unique_codes[BATCH_SIZE:]
            st_names = [st["name"] for st in validated_sub_themes]
            assign_batch = hierarchy_assign_batch()
            if st_names:
                for chunk_idx, chunk_start in enumerate(range(0, len(remaining), assign_batch)):
                    chunk = remaining[chunk_start : chunk_start + assign_batch]
                    try:
                        raw2 = llm_invoke_with_skill(
                            llm,
                            "hierarchy_construction",
                            assign_codes_to_subthemes_prompt(
                                label, st_names, chunk, research_question
                            ),
                        )
                        parsed2 = clean_and_parse_json(raw2)
                        assigned_chunk: set = set()
                        for st_name, st_codes in parsed2.get("assignments", {}).items():
                            if not isinstance(st_codes, list):
                                continue
                            for st in validated_sub_themes:
                                if st["name"] == st_name:
                                    valid = [c for c in st_codes if c in set(chunk)]
                                    st["codes"].extend(valid)
                                    assigned_chunk.update(valid)
                                    break
                        for c in parsed2.get("unassigned", []):
                            if isinstance(c, str) and c in chunk and c not in assigned_chunk:
                                validated_ungrouped.append(c)
                                assigned_chunk.add(c)
                        validated_ungrouped.extend(c for c in chunk if c not in assigned_chunk)
                    except Exception as e:
                        log_step("HIERARCHY_BATCH2_ERROR", f"Cluster {cid} chunk {chunk_idx}: {e}")
                        validated_ungrouped.extend(chunk)
            else:
                validated_ungrouped.extend(remaining)

        if (
            hierarchy_embed_drain_enabled()
            and len(validated_ungrouped) > EMBED_DRAIN_THRESHOLD
            and validated_sub_themes
        ):
            try:
                validated_ungrouped = drain_ungrouped_to_subthemes(
                    validated_sub_themes, validated_ungrouped
                )
            except Exception as e:
                log_step("HIERARCHY_EMBED_DRAIN_ERROR", f"Cluster {cid}: {e}")

        return cid, {
            "label": label,
            "sub_themes": validated_sub_themes,
            "ungrouped_codes": validated_ungrouped,
        }

    pending = {
        cid: (codes, codebook.get(cid, f"Cluster {cid}"))
        for cid, codes in sorted(cluster_to_codes.items(), key=lambda x: int(x[0]))
        if cid not in hierarchy
    }

    _hier_lock = threading.Lock()
    workers = int(os.environ.get("GT_HIERARCHY_WORKERS", "8"))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_process_one, cid, codes, label): cid
            for cid, (codes, label) in pending.items()
        }
        for fut in as_completed(futures):
            cid, result = fut.result()
            with _hier_lock:
                hierarchy[cid] = result
                total_codes = sum(len(st["codes"]) for st in result["sub_themes"]) + len(
                    result["ungrouped_codes"]
                )
                log_step(
                    "HIERARCHY_CLUSTER_DONE",
                    f"Cluster {cid} ({result['label']}): "
                    f"{len(result['sub_themes'])} sub-themes, {total_codes} codes",
                )
                save_hierarchy(hierarchy)

    summary = f"Hierarchy: {len(hierarchy)} clusters with sub-theme groupings. See {display_path(HIERARCHY_PATH)}"
    return summary


@tool
def tree_assembly(research_question: str = "") -> str:
    """
    Merge meta-themes + hierarchy into gt_global_graph.json (tree + flat edges).
    Runs maybe_refine_hierarchy (env GT_HIERARCHY_REFINE) and may rewrite gt_hierarchy.json.
    """
    if not os.path.isfile(str(META_THEMES_PATH)):
        return json.dumps(
            {"error": "Missing gt_meta_themes.json; run meta_theme_grouping step first."}
        )
    if not os.path.isfile(str(HIERARCHY_PATH)):
        return json.dumps({"error": "Missing gt_hierarchy.json; run hierarchy step first."})
    codebook_path = str(CODEBOOK_PATH)
    if not os.path.isfile(codebook_path):
        return json.dumps({"error": "Missing codebook.json; run high-level step first."})

    with open(META_THEMES_PATH, encoding="utf-8") as f:
        meta_data = json.load(f)
    with open(HIERARCHY_PATH, encoding="utf-8") as f:
        hierarchy = json.load(f)

    def _invoke_skill(skill_key: str, human_prompt: str, **labels: Any) -> str:
        return llm_invoke_with_skill(llm, skill_key, human_prompt, **labels)

    hierarchy = maybe_refine_hierarchy(hierarchy, research_question or "", _invoke_skill)
    ensure_output_dirs()
    with open(HIERARCHY_PATH, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2)

    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)
    codebook = cb_data.get("codebook", {})

    meta_themes = meta_data.get("meta_themes", [])

    root_name = research_question or "Thematic Analysis"
    tree: Dict[str, Any] = {"name": root_name, "type": "root", "children": []}

    edges: List[Dict[str, str]] = []
    all_nodes: List[str] = [root_name]

    for mt in meta_themes:
        mt_name = mt.get("name", "Unnamed Meta-Theme")
        mt_node: Dict[str, Any] = {"name": mt_name, "type": "meta_theme", "children": []}
        edges.append({"parent": root_name, "child": mt_name})
        all_nodes.append(mt_name)

        for cid in mt.get("cluster_ids", []):
            cid = str(cid)
            cluster_label = codebook.get(cid, f"Cluster {cid}")
            cluster_entry = hierarchy.get(cid, {})
            label = cluster_entry.get("label", cluster_label)

            theme_node: Dict[str, Any] = {"name": label, "type": "theme", "children": []}
            edges.append({"parent": mt_name, "child": label})
            all_nodes.append(label)

            for st in cluster_entry.get("sub_themes", []):
                if isinstance(st, dict):
                    theme_node["children"].append(build_sub_theme_node(st, label, edges, all_nodes))

            for code in cluster_entry.get("ungrouped_codes", []):
                theme_node["children"].append({"name": code, "type": "code"})
                edges.append({"parent": label, "child": code})
                all_nodes.append(code)

            mt_node["children"].append(theme_node)

        tree["children"].append(mt_node)

    canonical_nodes = sorted(set(all_nodes))

    out = {
        "tree": tree,
        "canonical_nodes": canonical_nodes,
        "merge_groups": [],
        "edges": edges,
        "inferred_edges": [],
    }
    ensure_output_dirs()
    with open(GLOBAL_GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    summary = (
        f"Tree assembled: {len(canonical_nodes)} nodes, {len(edges)} edges (strict hierarchy). "
        f"See {display_path(GLOBAL_GRAPH_PATH)} (hierarchy may have been refined; {display_path(HIERARCHY_PATH)})"
    )
    return summary
