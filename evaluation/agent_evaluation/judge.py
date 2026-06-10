"""Judge / adjudicator: decides upheld / rejected / unclear for each challenge."""
from __future__ import annotations

import logging

from .llm_client import call_llm, parse_json_response
from .prompts import JUDGE_SYSTEM, JUDGE_USER_TEMPLATE
from .schemas import AnswererOutput, ChallengeOutput, CodeEntry, ComponentType, JudgeOutput
from .benchmarker import _format_criteria, _build_numbered_evidence

logger = logging.getLogger(__name__)


def adjudicate(
    code: CodeEntry,
    component: ComponentType,
    challenge: ChallengeOutput,
    answerer: AnswererOutput,
    model: str,
    base_url: str = "http://localhost:8000/v1",
) -> JudgeOutput:
    evidence_text, _ = _build_numbered_evidence(code, component=component)

    user = JUDGE_USER_TEMPLATE.format(
        label=code.label,
        definition=code.definition,
        inclusion_text=_format_criteria(code.inclusion),
        exclusion_text=_format_criteria(code.exclusion),
        evidence_text=evidence_text,
        component=component,
        challenge_question=challenge.challenge_question,
        critique_claim=challenge.critique_claim,
        failure_mode=challenge.failure_mode,
        failure_mechanism=challenge.failure_mechanism,
        evidence_used=challenge.evidence_used,
        evidence_disposition=answerer.evidence_disposition or "(not stated)",
        supporting_criterion=answerer.supporting_criterion or "(not stated)",
        criterion_location=answerer.criterion_location or "none",
        evidence_to_criteria_mapping=answerer.evidence_to_criteria_mapping or "(not stated)",
        answerer_response=answerer.response or "(no response provided)",
    )
    try:
        raw = call_llm(JUDGE_SYSTEM, user, model=model, base_url=base_url, temperature=0.0)
        data = parse_json_response(raw)
        decision = data.get("decision", "unclear").lower()
        if decision not in ("upheld", "rejected", "unclear"):
            decision = "unclear"

        evidence_role = data.get("evidence_role", "")
        author_resolution = data.get("author_resolution", "not_resolved").lower()
        if author_resolution not in ("resolved", "partially_resolved", "not_resolved"):
            author_resolution = "not_resolved"

        # Hard override: missing/invalid fields → force not_resolved
        # supporting_criterion_id is already "" if the model cited a non-existent ID
        if not answerer.evidence_disposition or not answerer.supporting_criterion_id:
            author_resolution = "not_resolved"

        # Hard override decision table — mirrors STEP 3 in judge prompt exactly:
        if evidence_role in ("background_only", "plausible_concern"):
            decision = "rejected"
        elif evidence_role == "demonstrated_failure":
            if author_resolution == "resolved":
                decision = "rejected"
            elif author_resolution == "not_resolved":
                decision = "upheld"
            elif author_resolution == "partially_resolved":
                decision = "unclear"

        return JudgeOutput(
            decision=decision,
            reasoning=data.get("reasoning", ""),
            evidence_role=evidence_role,
            failure_type=data.get("failure_type", ""),
            author_resolution=author_resolution,
            material_failure=bool(data.get("material_failure", False)),
        )
    except Exception as exc:
        logger.warning("Judge failed for %s/%s: %s", code.code_id, component, exc)
        return JudgeOutput(decision="unclear", reasoning=f"Judge error: {exc}")
