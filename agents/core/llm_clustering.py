"""HICode-style LLM axial clustering (alternative to embedding + K-means)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set

from .paths import DATA_DIR
from .utils import log_step

# Axial clustering: False = embedding + K-means (default); True = LLM (HICode-style).
USE_LLM_CLUSTERING = False


def use_llm_clustering() -> bool:
    return USE_LLM_CLUSTERING


def _llm_cluster_env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        n = int(raw)
    except ValueError:
        n = default
    return max(lo, min(hi, n))


def _llm_cluster_save_iters() -> bool:
    return os.environ.get("GT_LLM_CLUSTER_SAVE_ITER", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _load_codes_per_review_as_is(out_dir: str) -> List[List]:
    """Copy codes_per_review from gt_codes_only.json without dedup remapping."""
    codes_per_review: List[List] = []
    codes_only_path = os.path.join(out_dir, "gt_codes_only.json")
    if not os.path.isfile(codes_only_path):
        return codes_per_review
    try:
        with open(codes_only_path, encoding="utf-8") as f:
            codes_only_data = json.load(f)
        for item in codes_only_data.get("codes_per_review", []):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            review_id, review_codes = item[0], item[1]
            if not isinstance(review_codes, list):
                continue
            row = [c if isinstance(c, str) else str(c) for c in review_codes]
            codes_per_review.append([review_id, row])
    except (json.JSONDecodeError, OSError):
        pass
    return codes_per_review


def _llm_cluster_context_length() -> int:
    """SGLang ``--context-length``; must match ``launch_sgl.sh`` (``GT_SGLANG_CONTEXT_LENGTH``)."""
    return _llm_cluster_env_int("GT_SGLANG_CONTEXT_LENGTH", 16384, 2048, 131072)


def _llm_cluster_max_completion() -> int:
    """Upper bound on completion tokens per clustering call (HICode uses 8192)."""
    return _llm_cluster_env_int("GT_LLM_CLUSTER_MAX_COMPLETION", 8192, 256, 32768)


def _llm_cluster_completion_budget(system_prompt: str, batch: List[str]) -> int:
    """Completion tokens that fit in the server context window alongside this batch.

    SGLang rejects requests when ``input_tokens + max_completion_tokens > context_length``.
    We estimate input size conservatively (chars/3) and cap completion accordingly.
    """
    ctx = _llm_cluster_context_length()
    cap = _llm_cluster_max_completion()
    input_est = (len(system_prompt) + len(str(batch))) // 3 + 128
    reserve = 256
    return min(cap, max(512, ctx - input_est - reserve))


def _hicode_run_batch(
    llm: Any, system_prompt: str, batch: List[str], iteration: int, b: int
) -> Dict[str, Any]:
    """One clustering call, mirroring HICode's ``_run_batch``.

    System message is the clustering prompt; the user message is the Python-repr list of
    labels (exactly as HICode sends ``str(labels[...])``). Returns the parsed JSON dict,
    or ``{}`` on a parse failure -- HICode likewise skips a batch whose output won't
    parse, letting those labels fall away. ``remove_think_tags`` + ``clean_and_parse_json``
    are the only local-model adaptations (the served model can emit thinking blocks /
    code fences; HICode used an API with response_format=json_object instead).
    """
    import time

    from langchain_core.messages import HumanMessage, SystemMessage

    from .utils import clean_and_parse_json, record_llm_usage, remove_think_tags

    t0 = time.monotonic()
    max_tokens = _llm_cluster_completion_budget(system_prompt, batch)
    log_step(
        "LLM_CLUSTER_BUDGET",
        f"iter={iteration} batch={b} labels={len(batch)} "
        f"completion_budget={max_tokens} context={_llm_cluster_context_length()}",
    )
    ai = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=str(batch))],
        max_tokens=max_tokens,
    )
    record_llm_usage(
        "llm_clustering",
        ai,
        latency_ms=(time.monotonic() - t0) * 1000.0,
        labels={"phase": "llm_clustering", "iteration": iteration, "batch": b},
    )
    raw = getattr(ai, "content", "") or ""
    try:
        parsed = clean_and_parse_json(remove_think_tags(raw))
    except (ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _hicode_build_mapping(
    cluster_list: List[Dict[str, List[str]]],
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Reproduce HICode's ``metrics.create_mapping``.

    Chains each original label through every clustering iteration to its final
    top-level cluster. A label whose cluster is not carried into the next round is
    *abandoned* (dropped), exactly as in HICode. Matching is on lowercased strings.

    Returns ``(label_lower -> final_key_lower, key_lower -> display_name)``.
    """
    reverses: List[Dict[str, str]] = []
    key_display: Dict[str, str] = {}
    for cluster in cluster_list:
        rev: Dict[str, str] = {}
        for k, members in cluster.items():
            kl = k.strip().lower()
            key_display[kl] = k.strip()
            for v in members:
                if isinstance(v, str):
                    rev[v.strip().lower()] = kl
        reverses.append(rev)

    if not reverses:
        return {}, key_display

    generated = dict(reverses[0])
    for i in range(1, len(reverses)):
        for label in list(generated.keys()):
            cur_key = generated[label]
            if cur_key in reverses[i]:
                generated[label] = reverses[i][cur_key]
            else:
                generated.pop(label)
    return generated, key_display


def _assign_integer_cluster_ids(
    leaf_cluster: Dict[str, List[str]],
) -> tuple[Dict[str, List[str]], Dict[str, str]]:
    """Sort themes by size; return cluster_to_codes and cluster_theme_names."""
    sorted_items = sorted(
        leaf_cluster.items(),
        key=lambda kv: (-len(kv[1]), kv[0].lower()),
    )
    cluster_to_codes: Dict[str, List[str]] = {}
    cluster_theme_names: Dict[str, str] = {}
    for i, (theme, codes) in enumerate(sorted_items):
        cid = str(i)
        cluster_to_codes[cid] = list(codes)
        label = theme.strip()
        if not label or len(label) > 120:
            label = f"Cluster {cid}"
        cluster_theme_names[cid] = label
    return cluster_to_codes, cluster_theme_names


def axial_llm_cluster(
    all_codes: List[str],
    research_question: str,
    llm: Any,
    out_dir: str = str(DATA_DIR),
) -> str:
    """HICode-style multi-iteration LLM clustering; writes gt_clustered_codes.json.

    Faithful port of HICode's ``cluster_labels_gpt`` + ``metrics.create_mapping``:
    the unique input labels are clustered in batches, the resulting cluster names are
    re-clustered for up to ``max_iter`` rounds, then every original label is chained to
    its final theme. Labels the model omits, or whose cluster is not carried forward,
    are dropped -- HICode's ``abandoned_labels`` behavior (no rescue / no coverage repair).
    """
    from .prompts import llm_clustering_prompt

    if not all_codes:
        return "No codes to cluster."

    input_codes = [c.strip() for c in all_codes if isinstance(c, str) and c.strip()]
    if not input_codes:
        return "No codes to cluster."

    # HICode hardcodes batch_size=100 and max_n_iter=3; kept as env knobs with those defaults.
    max_iter = _llm_cluster_env_int("GT_LLM_CLUSTER_MAX_ITER", 3, 1, 10)
    batch_size = _llm_cluster_env_int("GT_LLM_CLUSTER_BATCH_SIZE", 100, 1, 200)
    system_prompt = llm_clustering_prompt(research_question)

    # HICode process_labels clusters the *unique* labels (it uses list(set(...)));
    # sorted() keeps that uniqueness but makes batch composition reproducible.
    labels_to_cluster: List[str] = sorted(set(input_codes))
    cluster_list: List[Dict[str, List[str]]] = []

    for iteration in range(max_iter):
        n_labels = len(labels_to_cluster)
        n_batch = (n_labels + batch_size - 1) // batch_size if n_labels else 0
        log_step(
            "LLM_CLUSTER_ITER",
            f"iteration={iteration} labels={n_labels} batches={n_batch}",
        )

        cluster: Dict[str, List[str]] = {}
        for b in range(n_batch):
            batch = labels_to_cluster[b * batch_size : (b + 1) * batch_size]
            if not batch:
                continue
            model_output = _hicode_run_batch(llm, system_prompt, batch, iteration, b)
            for k, members in model_output.items():
                if isinstance(members, list):
                    cluster.setdefault(str(k), []).extend(m for m in members if isinstance(m, str))

        cluster_list.append(cluster)

        if _llm_cluster_save_iters():
            iter_dir = os.path.join(out_dir, "llm_clustering")
            os.makedirs(iter_dir, exist_ok=True)
            with open(
                os.path.join(iter_dir, f"cluster_iter_{iteration}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(cluster, f, indent=2)

        labels_to_cluster = list(cluster.keys())
        if n_batch <= 1:
            break

    # HICode create_mapping: chain each original label to its final theme; drop abandoned.
    final_map, key_display = _hicode_build_mapping(cluster_list)
    orig_display: Dict[str, str] = {}
    for c in input_codes:
        orig_display.setdefault(c.strip().lower(), c)

    theme_to_codes: Dict[str, List[str]] = {}
    seen_codes: Set[str] = set()
    for label_lower, final_key_lower in final_map.items():
        code = orig_display.get(label_lower)
        if code is None or code in seen_codes:
            continue
        theme = key_display.get(final_key_lower, final_key_lower)
        theme_to_codes.setdefault(theme, []).append(code)
        seen_codes.add(code)

    n_dropped = len(set(input_codes)) - len(seen_codes)
    log_step(
        "LLM_CLUSTER_DROPPED",
        f"{n_dropped}/{len(set(input_codes))} unique labels abandoned "
        "(HICode drops labels not chained through all iterations).",
    )

    if not theme_to_codes:
        return "LLM clustering produced no usable clusters."

    cluster_to_codes, cluster_theme_names = _assign_integer_cluster_ids(theme_to_codes)

    all_codes_out: List[str] = []
    labels: List[int] = []
    for cid, codes in sorted(cluster_to_codes.items(), key=lambda x: int(x[0])):
        for code in codes:
            all_codes_out.append(code)
            labels.append(int(cid))

    codes_per_review = _load_codes_per_review_as_is(out_dir)

    out: Dict[str, Any] = {
        "clustering_method": "llm",
        "cluster_theme_names": cluster_theme_names,
        "k": len(cluster_to_codes),
        "cluster_to_codes": cluster_to_codes,
        "all_codes": all_codes_out,
        "labels": labels,
    }
    if codes_per_review:
        out["codes_per_review"] = codes_per_review

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gt_clustered_codes.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = [
        f"Axial coding (LLM clustering): {len(cluster_to_codes)} clusters, "
        f"{len(cluster_list)} iteration(s).",
        "",
    ]
    for cid, codes in sorted(cluster_to_codes.items(), key=lambda x: int(x[0])):
        theme = cluster_theme_names.get(cid, f"Cluster {cid}")
        lines.append(f"Cluster {cid} ({theme}):")
        for c in codes[:20]:
            lines.append(f"  - {c}")
        if len(codes) > 20:
            lines.append(f"  - ... and {len(codes) - 20} more")
        lines.append("")
    return "\n".join(lines).strip()
