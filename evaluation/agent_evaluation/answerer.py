"""Answerer: the codebook generator model defends its component against a challenge."""
from __future__ import annotations

import logging

from .llm_client import call_llm, parse_json_response
from .prompts import ANSWERER_SYSTEM, ANSWERER_USER_TEMPLATE
from .schemas import AnswererOutput, ChallengeOutput, CodeEntry, ComponentType
from .benchmarker import (
    _build_numbered_evidence,
    _format_criteria_with_ids,
    build_criterion_id_map,
)

logger = logging.getLogger(__name__)

_LOCATION_FOR_PREFIX = {"INC": "inclusion", "EXC": "exclusion", "LABEL": "label", "DEF": "definition"}


def answer_challenge(
    code: CodeEntry,
    component: ComponentType,
    challenge: ChallengeOutput,
    model: str,
    base_url: str = "http://localhost:8000/v1",
) -> AnswererOutput:
    evidence_text, _ = _build_numbered_evidence(code, component=component)
    inclusion_text, _ = _format_criteria_with_ids(code.inclusion, "INC")
    exclusion_text, _ = _format_criteria_with_ids(code.exclusion, "EXC")

    # Build the full valid ID set for this code entry
    valid_ids = build_criterion_id_map(code)  # {LABEL, DEF, INC-1…, EXC-1…}
    valid_id_list = ", ".join(sorted(valid_ids.keys()))

    user = ANSWERER_USER_TEMPLATE.format(
        label=code.label,
        definition=code.definition,
        inclusion_text=inclusion_text,
        exclusion_text=exclusion_text,
        evidence_text=evidence_text,
        component=component,
        challenge_question=challenge.challenge_question,
        critique_claim=challenge.critique_claim,
        failure_mode=challenge.failure_mode,
        failure_mechanism=challenge.failure_mechanism,
        evidence_used=challenge.evidence_used,
        valid_criterion_ids=valid_id_list,
    )
    try:
        raw = call_llm(ANSWERER_SYSTEM, user, model=model, base_url=base_url, temperature=0.3)
        data = parse_json_response(raw)

        disposition = data.get("evidence_disposition", "").lower().strip()
        if disposition not in ("included", "excluded", "undecidable"):
            disposition = ""

        # Exact ID validation — only accept IDs that exist in this codebook entry
        cid_raw = (data.get("supporting_criterion_id") or "").strip().upper()
        if cid_raw not in valid_ids:
            if cid_raw:
                logger.info(
                    "Answerer: invalid criterion ID '%s' for %s/%s — cleared",
                    cid_raw, code.code_id, component,
                )
            cid_raw = ""

        # Derive criterion_location from the ID prefix
        criterion_loc = "none"
        if cid_raw:
            prefix = cid_raw.split("-")[0]  # INC, EXC, LABEL, DEF
            criterion_loc = _LOCATION_FOR_PREFIX.get(prefix, "none")

        # Resolved criterion text (looked up from actual codebook — not from model output)
        supporting_text = valid_ids.get(cid_raw, "")

        return AnswererOutput(
            response=data.get("response", ""),
            evidence_disposition=disposition,
            supporting_criterion=supporting_text,   # always real codebook text
            supporting_criterion_id=cid_raw,        # the ID the model cited
            criterion_location=criterion_loc,
            evidence_to_criteria_mapping=data.get("evidence_to_criteria_mapping", ""),
        )
    except Exception as exc:
        logger.warning("Answerer failed for %s/%s: %s", code.code_id, component, exc)
        return AnswererOutput(
            response="",
            evidence_disposition="",
            supporting_criterion="",
            supporting_criterion_id="",
            criterion_location="none",
            evidence_to_criteria_mapping="",
        )
