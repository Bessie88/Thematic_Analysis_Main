"""
CRB-style evaluation pipeline — phased execution.

Five phases, each loads one model at a time:
  Phase 1 (benchmark): Gemma 3 27B  → challenges.jsonl
  Phase 2 (gate):      Mistral Small 24B → gated.jsonl
  Phase 3 (answer):    Qwen3.6 (same model that generated codebook) → answered.jsonl
  Phase 4 (judge):     Qwen3-30B → eval_outcomes.jsonl
  Phase 5 (bt):        no LLM → bt_results.json
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .answerer import answer_challenge
from .benchmarker import generate_challenge
from .gate import check_challenge
from .judge import adjudicate
from .schemas import COMPONENTS, CodeEntry, EvalItem

logger = logging.getLogger(__name__)


# ── Model config ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    model: str
    base_url: str


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_review_evidence(md_path: Path) -> dict[int, list[dict]]:
    """Parse gt_open_codes_all_reviews.md into {review_idx: [{code, evidence}]}."""
    if not md_path.is_file():
        return {}
    text = md_path.read_text(encoding="utf-8")
    blocks = re.split(r'## Review (\d+)', text)
    parsed: dict[int, list[dict]] = {}
    for i in range(1, len(blocks), 2):
        rev_idx = int(blocks[i])
        content = blocks[i + 1]
        items = []
        for m in re.finditer(r'- Code: (.+?)\n\s+Evidence: "(.+?)"', content, re.DOTALL):
            items.append({"code": m.group(1).strip(), "evidence": m.group(2).strip()})
        parsed[rev_idx] = items
    return parsed


def _source_evidence_for_open_codes(
    open_code_names: set[str],
    review_evidence: dict[int, list[dict]],
    codes_per_review: list,
    max_snippets: int = 20,
) -> list[str]:
    """
    Find review evidence snippets that contain any of the given open codes.
    Returns formatted strings: '[Review N | <code>] "<evidence>"'
    """
    snippets: list[str] = []
    seen: set[str] = set()

    for item in codes_per_review:
        rev_idx, rev_codes = item[0], item[1]
        matching = open_code_names & set(rev_codes)
        if not matching:
            continue
        for entry in review_evidence.get(rev_idx, []):
            if entry["code"] not in open_code_names:
                continue
            text = f'[Review {rev_idx} | {entry["code"]}] "{entry["evidence"]}"'
            if text not in seen:
                seen.add(text)
                snippets.append(text)
        if len(snippets) >= max_snippets:
            break

    return snippets[:max_snippets]


def _entry_to_code(
    entry: dict,
    code_id: str,
    level: str,
    open_code_names: set[str],
    review_evidence: dict[int, list[dict]],
    codes_per_review: list,
    neighbor_open_codes: set[str] | None = None,
) -> CodeEntry:
    # Examples already embedded in inclusion/exclusion criteria
    embedded: list[str] = []
    for c in entry.get("inclusion", []) + entry.get("exclusion", []):
        embedded.extend(c.get("examples", []))

    # Source evidence retrieved from original reviews via open-code linkage
    source_ev = _source_evidence_for_open_codes(
        open_code_names, review_evidence, codes_per_review, max_snippets=20
    )

    # Combine: embedded examples first (most curated), then source evidence
    all_ev = list(dict.fromkeys(embedded + source_ev))[:40]

    # Neighbor evidence: evidence from other clusters (used for exclusion challenges)
    neighbor_ev: list[str] = []
    if neighbor_open_codes:
        neighbor_ev = _source_evidence_for_open_codes(
            neighbor_open_codes, review_evidence, codes_per_review, max_snippets=20
        )
        # Remove any overlap with this code's own evidence
        neighbor_ev = [e for e in neighbor_ev if e not in set(all_ev)]

    return CodeEntry(
        code_id=code_id,
        label=entry.get("label", ""),
        definition=entry.get("definition", ""),
        inclusion=entry.get("inclusion", []),
        exclusion=entry.get("exclusion", []),
        level=level,
        evidence_snippets=all_ev,
        neighbor_evidence_snippets=neighbor_ev,
    )


def load_codes(run_dir: Path) -> list[CodeEntry]:
    """Load codes from codebook_enriched (cluster) and meta_themes_enriched,
    enriching each with source review evidence via open-code linkage."""
    data_dir = run_dir / "data"
    cb_path  = data_dir / "codebook.json"
    mt_path  = data_dir / "gt_meta_themes_enriched.json"
    cl_path  = data_dir / "gt_clustered_codes.json"
    md_path  = data_dir / "gt_open_codes_all_reviews.md"

    # Parse review evidence once
    review_evidence  = _parse_review_evidence(md_path)

    # Load cluster → open codes mapping and review → codes mapping
    cluster_to_codes: dict[str, list[str]] = {}
    codes_per_review: list = []
    if cl_path.is_file():
        with open(cl_path, encoding="utf-8") as f:
            cl_data = json.load(f)
        cluster_to_codes = {str(k): v for k, v in cl_data.get("cluster_to_codes", {}).items()}
        codes_per_review = cl_data.get("codes_per_review", [])
    else:
        logger.warning("gt_clustered_codes.json not found in %s", data_dir)

    codes: list[CodeEntry] = []

    # ── Cluster-level codes ───────────────────────────────────────────────────
    if cb_path.is_file():
        with open(cb_path, encoding="utf-8") as f:
            cb_data = json.load(f)

        # Pre-compute all open codes across all clusters for neighbor lookup
        all_cluster_open_codes: dict[str, set[str]] = {
            str(idx): set(names)
            for idx, names in cluster_to_codes.items()
        }
        all_open_codes_union: set[str] = set().union(*all_cluster_open_codes.values()) if all_cluster_open_codes else set()

        for code_idx, entry in cb_data.get("codebook_enriched", {}).items():
            if not isinstance(entry, dict):
                continue
            open_codes = set(cluster_to_codes.get(str(code_idx), []))
            neighbor_open_codes = all_open_codes_union - open_codes
            codes.append(_entry_to_code(
                entry, f"cluster_{code_idx}", "cluster",
                open_codes, review_evidence, codes_per_review,
                neighbor_open_codes=neighbor_open_codes,
            ))
    else:
        logger.warning("codebook.json not found in %s", data_dir)

    # ── Meta-theme-level codes ────────────────────────────────────────────────
    if mt_path.is_file():
        with open(mt_path, encoding="utf-8") as f:
            mt_data = json.load(f)
        for mt_idx, entry in enumerate(mt_data.get("meta_themes_enriched", [])):
            if not isinstance(entry, dict):
                continue
            if not entry.get("inclusion") and not entry.get("exclusion"):
                logger.warning("Meta-theme %d has no inclusion/exclusion — skipping", mt_idx)
                continue
            # Meta-theme spans multiple clusters → union of all their open codes
            cluster_ids = [str(c) for c in entry.get("cluster_ids", [])]
            open_codes: set[str] = set()
            for cid in cluster_ids:
                open_codes.update(cluster_to_codes.get(cid, []))
            codes.append(_entry_to_code(
                entry, f"meta_{mt_idx}", "meta_theme",
                open_codes, review_evidence, codes_per_review,
            ))
    else:
        logger.warning("gt_meta_themes_enriched.json not found in %s", data_dir)

    logger.info(
        "Loaded %d codes from %s (%d cluster, %d meta_theme); "
        "%d reviews parsed for source evidence",
        len(codes), run_dir.name,
        sum(1 for c in codes if c.level == "cluster"),
        sum(1 for c in codes if c.level == "meta_theme"),
        len(review_evidence),
    )
    return codes


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows → %s", len(rows), path)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(items: list[EvalItem], out_path: Path) -> None:
    _write_jsonl([i.to_dict() for i in items], out_path)


# ── Phase 1: Challenge generation ─────────────────────────────────────────────

def phase_benchmark(
    runs_root: Path,
    run_names: list[str],
    generator_model: str,
    benchmarker_cfg: ModelConfig,
    out_path: Path,
    n_challenges: int = 1,
) -> None:
    """Generate n_challenges challenges per (run, code, component).

    Multiple challenges share the same item key (benchmarker, codebook_id,
    code_id, component), so they become repeated observations of the same item
    in the BT model — improving item difficulty estimates.
    """
    rows: list[dict] = []

    for run_name in run_names:
        run_dir = runs_root / run_name
        if not run_dir.is_dir():
            logger.warning("Missing run dir: %s", run_dir)
            continue

        gen_model   = _generator_model_from_name(run_name, generator_model)
        codebook_id = run_name
        codes = load_codes(run_dir)
        logger.info(
            "Phase-benchmark %s: %d codes × 4 components × %d challenges",
            run_name, len(codes), n_challenges,
        )

        for code in codes:
            for component in COMPONENTS:
                for _ in range(n_challenges):
                    challenge = generate_challenge(
                        code, component,
                        benchmarker_cfg.model, benchmarker_cfg.base_url,
                    )
                    row: dict = {
                        "generator_model":   gen_model,
                        "benchmarker_model": benchmarker_cfg.model,
                        "codebook_id":       codebook_id,
                        "code_id":           code.code_id,
                        "level":             code.level,
                        "component":         component,
                        "_label":      code.label,
                        "_definition": code.definition,
                        "_inclusion":  code.inclusion,
                        "_exclusion":  code.exclusion,
                        "_evidence":   code.evidence_snippets,
                    }
                    if challenge is not None:
                        row.update({
                            "challenge_question":  challenge.challenge_question,
                            "critique_claim":      challenge.critique_claim,
                            "evidence_used":       challenge.evidence_used,
                            "failure_mode":        challenge.failure_mode,
                            "failure_mechanism":   challenge.failure_mechanism,
                            "status":              "pending_gate",
                        })
                    else:
                        row.update({
                            "challenge_question": "",
                            "critique_claim":     "",
                            "evidence_used":      "",
                            "failure_mode":       "",
                            "failure_mechanism":  "",
                            "status":             "dropped",
                            "drop_reason":        "benchmarker returned None",
                        })
                    rows.append(row)

    _write_jsonl(rows, out_path)


# ── Phase 2: Feasibility gate ─────────────────────────────────────────────────

def phase_gate(
    challenges_path: Path,
    gate_cfg: ModelConfig,
    out_path: Path,
    max_retries: int = 2,
) -> None:
    """Mistral Small 24B checks each challenge. Dropped if still invalid after retries."""
    from .schemas import ChallengeOutput
    rows = _read_jsonl(challenges_path)
    out_rows: list[dict] = []

    for row in rows:
        if row.get("status") == "dropped":
            out_rows.append(row)
            continue

        component = row["component"]
        challenge = ChallengeOutput(
            challenge_question=row["challenge_question"],
            critique_claim=row["critique_claim"],
            evidence_used=row["evidence_used"],
            failure_mode=row.get("failure_mode", "unsupported_or_vague"),
            failure_mechanism=row.get("failure_mechanism", ""),
        )
        gate_decision = "invalid"
        gate_reason   = ""

        code = _code_from_row(row)
        for attempt in range(1 + max_retries):
            gate = check_challenge(challenge, component, gate_cfg.model, gate_cfg.base_url, code=code)
            gate_decision = gate.decision
            gate_reason   = gate.reason
            if gate_decision == "valid":
                break
            logger.info(
                "Gate invalid (attempt %d/%d) %s/%s: %s",
                attempt + 1, 1 + max_retries,
                row["codebook_id"], component, gate_reason,
            )

        out_row = dict(row)
        if gate_decision == "valid":
            out_row["status"]               = "pending_answer"
            out_row["feasibility_decision"] = "valid"
            out_row["gate_reason"]          = gate_reason
        else:
            out_row["status"]               = "dropped"
            out_row["feasibility_decision"] = "invalid"
            out_row["drop_reason"]          = gate_reason
        out_rows.append(out_row)

    _write_jsonl(out_rows, out_path)


# ── Phase 3: Answerer ─────────────────────────────────────────────────────────

def phase_answer(
    gated_path: Path,
    answerer_cfg: ModelConfig,
    out_path: Path,
) -> None:
    """Qwen3.6 defends each valid challenge. Same model that generated the codebook."""
    from .schemas import ChallengeOutput
    rows = _read_jsonl(gated_path)
    out_rows: list[dict] = []

    for row in rows:
        out_row = dict(row)

        if row.get("status") != "pending_answer":
            out_rows.append(out_row)
            continue

        code = _code_from_row(row)
        challenge = ChallengeOutput(
            challenge_question=row["challenge_question"],
            critique_claim=row["critique_claim"],
            evidence_used=row["evidence_used"],
            failure_mode=row.get("failure_mode", "unsupported_or_vague"),
            failure_mechanism=row.get("failure_mechanism", ""),
        )
        answerer_out = answer_challenge(
            code, row["component"], challenge,
            answerer_cfg.model, answerer_cfg.base_url,
        )
        out_row["answerer_model"] = answerer_cfg.model
        out_row["answerer_response"] = answerer_out.response
        out_row["answerer_evidence_disposition"] = answerer_out.evidence_disposition
        out_row["answerer_supporting_criterion_id"] = answerer_out.supporting_criterion_id
        out_row["answerer_supporting_criterion"] = answerer_out.supporting_criterion
        out_row["answerer_criterion_location"] = answerer_out.criterion_location
        out_row["answerer_evidence_mapping"] = answerer_out.evidence_to_criteria_mapping
        out_row["status"] = "pending_judge"
        out_rows.append(out_row)

    _write_jsonl(out_rows, out_path)


# ── Phase 4: Judge ────────────────────────────────────────────────────────────

def phase_judge(
    answered_path: Path,
    judge_cfg: ModelConfig,
    out_path: Path,
) -> None:
    """Qwen3-30B adjudicates each challenge given the answerer's response."""
    from .schemas import AnswererOutput, ChallengeOutput
    rows = _read_jsonl(answered_path)
    items: list[EvalItem] = []

    for row in rows:
        component = row["component"]

        if row.get("status") != "pending_judge":
            items.append(EvalItem(
                generator_model=row["generator_model"],
                answerer_model=row.get("answerer_model", row["generator_model"]),
                benchmarker_model=row["benchmarker_model"],
                codebook_id=row["codebook_id"],
                code_id=row["code_id"],
                level=row.get("level", "cluster"),
                component=component,
                challenge_question=row.get("challenge_question", ""),
                critique_claim=row.get("critique_claim", ""),
                failure_mode=row.get("failure_mode", ""),
                failure_mechanism=row.get("failure_mechanism", ""),
                evidence_used=row.get("evidence_used", ""),
                feasibility_decision=row.get("feasibility_decision", "dropped"),
                answerer_response=row.get("answerer_response"),
                answerer_evidence_disposition=row.get("answerer_evidence_disposition"),
                answerer_supporting_criterion_id=row.get("answerer_supporting_criterion_id"),
                answerer_supporting_criterion=row.get("answerer_supporting_criterion"),
                answerer_criterion_location=row.get("answerer_criterion_location"),
                answerer_evidence_mapping=row.get("answerer_evidence_mapping"),
                judge_decision=None,
                judge_reasoning=None,
                judge_evidence_role=None,
                judge_failure_type=None,
                judge_author_resolution=None,
                judge_material_failure=None,
                outcome_y=None,
                drop_reason=row.get("drop_reason", ""),
            ))
            continue

        code = _code_from_row(row)
        challenge = ChallengeOutput(
            challenge_question=row["challenge_question"],
            critique_claim=row["critique_claim"],
            evidence_used=row["evidence_used"],
            failure_mode=row.get("failure_mode", "unsupported_or_vague"),
            failure_mechanism=row.get("failure_mechanism", ""),
        )
        answerer = AnswererOutput(
            response=row.get("answerer_response", ""),
            evidence_disposition=row.get("answerer_evidence_disposition", ""),
            supporting_criterion=row.get("answerer_supporting_criterion", ""),
            supporting_criterion_id=row.get("answerer_supporting_criterion_id", ""),
            criterion_location=row.get("answerer_criterion_location", "none"),
            evidence_to_criteria_mapping=row.get("answerer_evidence_mapping", ""),
        )

        verdict = adjudicate(
            code, component, challenge, answerer,
            judge_cfg.model, judge_cfg.base_url,
        )

        outcome_y: int | None
        if verdict.decision == "upheld":
            outcome_y = 0
        elif verdict.decision == "rejected":
            outcome_y = 1
        else:
            outcome_y = None

        items.append(EvalItem(
            generator_model=row["generator_model"],
            answerer_model=row.get("answerer_model", row["generator_model"]),
            benchmarker_model=row["benchmarker_model"],
            codebook_id=row["codebook_id"],
            code_id=row["code_id"],
            level=row.get("level", "cluster"),
            component=component,
            challenge_question=row["challenge_question"],
            critique_claim=row["critique_claim"],
            failure_mode=row.get("failure_mode", ""),
            failure_mechanism=row.get("failure_mechanism", ""),
            evidence_used=row["evidence_used"],
            feasibility_decision="valid",
            answerer_response=answerer.response,
            answerer_evidence_disposition=answerer.evidence_disposition,
            answerer_supporting_criterion_id=answerer.supporting_criterion_id,
            answerer_supporting_criterion=answerer.supporting_criterion,
            answerer_criterion_location=answerer.criterion_location,
            answerer_evidence_mapping=answerer.evidence_to_criteria_mapping,
            judge_decision=verdict.decision,
            judge_reasoning=verdict.reasoning,
            judge_evidence_role=verdict.evidence_role,
            judge_failure_type=verdict.failure_type,
            judge_author_resolution=verdict.author_resolution,
            judge_material_failure=verdict.material_failure,
            outcome_y=outcome_y,
            drop_reason=None if verdict.decision != "unclear" else "judge: unclear",
        ))

    write_jsonl(items, out_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _code_from_row(row: dict) -> CodeEntry:
    return CodeEntry(
        code_id=row["code_id"],
        label=row["_label"],
        definition=row["_definition"],
        inclusion=row["_inclusion"],
        exclusion=row["_exclusion"],
        level=row.get("level", "cluster"),
        evidence_snippets=row["_evidence"],
    )


def _generator_model_from_name(run_name: str, default: str) -> str:
    parts = run_name.rsplit("_run", 1)[0].rsplit("_", 1)
    return parts[-1] if len(parts) > 1 else default
