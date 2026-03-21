"""CLI entry point for the GT pipeline."""
import argparse
import json

from agents.core.paths import (
    CLUSTERED_CODES_PATH,
    CODEBOOK_PATH,
    DEFAULT_DATA_CSV,
    GLOBAL_GRAPH_PATH,
    GRAPH_PATH,
    GT_CODES_ONLY_PATH,
    HIERARCHY_PATH,
    OPEN_CODES_MARKDOWN_PATH,
    display_path,
    ensure_output_dirs,
)
from agents.core.utils import extract_codes, log_step
import pandas as pd
from agents.core.app import app


def _base_state(research_question: str) -> dict:
    """Initial state dict for the LangGraph; matches GTState shape."""
    return {
        "research_question": research_question,
        "raw_text": "",
        "open_codes": None,
        "open_codes_validation": None,
        "open_codes_validation_feedback": None,
        "_open_coding_retries": 0,
        "all_codes_for_axial": None,
        "axial_mapping": None,
        "_cluster_refinement_done": False,
        "codebook": None,
        "hierarchy": None,
        "graph": None,
        "global_graph": None,
        "tool_call": None,
        "step": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--open-coding-only", action="store_true", help="Run only open coding; save gt_codes_only.json and exit (so SGLang can be killed before axial).")
    p.add_argument("--axial-only", action="store_true", help="Load gt_codes_only.json and run only the axial step in the graph.")
    p.add_argument("--high-level-only", action="store_true", help="Load gt_clustered_codes.json and run high-level code generation (LLM must be up).")
    p.add_argument("--refine-only", action="store_true", help="Run high-level (if needed) then refine_cluster_assignments. Requires codebook.json and gt_clustered_codes.json. LLM must be up.")
    p.add_argument("--hierarchy-only", action="store_true", help="Run hierarchy construction (relationship classification + edges). LLM must be up.")
    p.add_argument("--graph-only", action="store_true", help="Run graph construction (transitivity inference). No LLM needed.")
    p.add_argument("--global-graph-only", action="store_true", help="Run global graph construction (merge clusters, optional cross-cluster linking). LLM needed unless --skip-cross-cluster.")
    p.add_argument("--skip-cross-cluster", action="store_true", help="Skip cross-cluster linking in global graph (LLM-free).")
    p.add_argument("--sim-threshold", type=float, default=0.6, help="Cosine similarity threshold for hierarchy pair filtering (default 0.6).")
    p.add_argument("--research-question", required=True, help="Research question that conditions the entire GT pipeline.")
    p.add_argument("--data", default=str(DEFAULT_DATA_CSV), help="CSV with review_text column.")
    args = p.parse_args()

    ensure_output_dirs()
    rq = args.research_question
    log_step("RESEARCH_QUESTION", rq)

    if args.high_level_only:
        if not CLUSTERED_CODES_PATH.is_file():
            print(f"Error: {display_path(CLUSTERED_CODES_PATH)} not found. Run axial step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "done"
        final_hl = app.invoke(state, config={"recursion_limit": 25})
        codebook = final_hl.get("codebook") or {}
        log_step("CODEBOOK_COMPLETE", f"Generated {len(codebook)} high-level labels. See {display_path(CODEBOOK_PATH)}")
        raise SystemExit(0)

    if args.refine_only:
        if not CODEBOOK_PATH.is_file():
            print(f"Error: {display_path(CODEBOOK_PATH)} not found. Run high-level step first.")
            raise SystemExit(1)
        if not CLUSTERED_CODES_PATH.is_file():
            print(f"Error: {display_path(CLUSTERED_CODES_PATH)} not found. Run axial step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "refine"
        with open(CODEBOOK_PATH, encoding="utf-8") as f:
            existing_cb = json.load(f)
        state["codebook"] = existing_cb.get("codebook", {})
        final_refine = app.invoke(state, config={"recursion_limit": 25})
        summary = final_refine.get("_cluster_refinement_done", False)
        log_step("REFINE_COMPLETE", "Cluster refinement finished." if summary else "Refine step completed.")
        raise SystemExit(0)

    if args.graph_only:
        if not HIERARCHY_PATH.is_file():
            print(f"Error: {display_path(HIERARCHY_PATH)} not found. Run hierarchy step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "graph"
        final_graph = app.invoke(state, config={"recursion_limit": 25})
        log_step("GRAPH_COMPLETE", final_graph.get("graph", ""))
        raise SystemExit(0)

    if args.global_graph_only:
        if not GRAPH_PATH.is_file():
            print(f"Error: {display_path(GRAPH_PATH)} not found. Run graph step first.")
            raise SystemExit(1)
        if not args.skip_cross_cluster and not CODEBOOK_PATH.is_file():
            print(f"Error: {display_path(CODEBOOK_PATH)} not found. Run high-level step first (or use --skip-cross-cluster).")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "global_graph"
        state["_sim_threshold"] = 0.7
        state["_skip_cross_cluster"] = args.skip_cross_cluster
        final_global = app.invoke(state, config={"recursion_limit": 25})
        log_step("GLOBAL_GRAPH_COMPLETE", final_global.get("global_graph", ""))
        raise SystemExit(0)

    if args.hierarchy_only:
        if not CODEBOOK_PATH.is_file():
            print(f"Error: {display_path(CODEBOOK_PATH)} not found. Run high-level step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "hierarchy"
        state["_sim_threshold"] = args.sim_threshold
        final_hier = app.invoke(state, config={"recursion_limit": 25})
        log_step("HIERARCHY_COMPLETE", final_hier.get("hierarchy", ""))
        raise SystemExit(0)

    if args.axial_only:
        with open(GT_CODES_ONLY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        state = _base_state(rq)
        state["all_codes_for_axial"] = data["all_codes"]
        final_axial = app.invoke(state, config={"recursion_limit": 25})
        axial_mapping = final_axial.get("axial_mapping", "")
        log_step("AXIAL_COMPLETE", axial_mapping[:500] + "..." if len(axial_mapping) > 500 else axial_mapping)
        raise SystemExit(0)

    review_text_df = pd.read_csv(args.data)
    reviews = review_text_df["review_text"].astype(str).tolist()
    all_open_codes = []

    for idx, review in enumerate(reviews, start=1):
        state = _base_state(rq)
        state["raw_text"] = review
        final_state = app.invoke(state, config={"recursion_limit": 25})
        all_open_codes.append((idx, final_state.get("open_codes", "")))

    with open(OPEN_CODES_MARKDOWN_PATH, "w", encoding="utf-8") as f:
        for review_id, codes in all_open_codes:
            f.write(f"## Review {review_id}\n\n{codes}\n\n")

    codes_per_review = [(review_id, extract_codes(raw)) for review_id, raw in all_open_codes]
    all_codes = [code for _, codes in codes_per_review for code in codes]
    with open(GT_CODES_ONLY_PATH, "w", encoding="utf-8") as f:
        json.dump({"all_codes": all_codes, "codes_per_review": [(rid, codes) for rid, codes in codes_per_review]}, f, indent=2)
    log_step("CODES_EXTRACTED", f"Total codes: {len(all_codes)} (from {len(codes_per_review)} reviews). See {display_path(GT_CODES_ONLY_PATH)}")

    if args.open_coding_only:
        log_step("OPEN_CODING_ONLY", "Exiting so SGLang can be killed before axial.")
        raise SystemExit(0)

    state = _base_state(rq)
    state["all_codes_for_axial"] = all_codes
    final_axial = app.invoke(state, config={"recursion_limit": 25})
    axial_mapping = final_axial.get("axial_mapping", "")
    log_step("AXIAL_COMPLETE", axial_mapping[:500] + "..." if len(axial_mapping) > 500 else axial_mapping)

    state = _base_state(rq)
    state["axial_mapping"] = "refine"
    final_refine = app.invoke(state, config={"recursion_limit": 25})
    codebook = final_refine.get("codebook") or {}
    log_step("CODEBOOK_REFINE_COMPLETE", f"High-level and refinement done. {len(codebook)} clusters. See {display_path(CODEBOOK_PATH)}")


if __name__ == "__main__":
    main()
