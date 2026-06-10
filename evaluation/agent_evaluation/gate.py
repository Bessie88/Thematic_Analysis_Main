"""Feasibility gate: checks whether a generated challenge is valid."""
from __future__ import annotations

import logging

from .llm_client import call_llm, parse_json_response
from .prompts import GATE_SYSTEM, GATE_USER_TEMPLATE, GATE_INSTRUCTIONS_CRITERIA, GATE_INSTRUCTIONS_CONCEPTUAL

CONCEPTUAL_COMPONENTS = {"label", "definition"}
from .schemas import ChallengeOutput, CodeEntry, ComponentType, FeasibilityOutput

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _evidence_grounded(challenge: ChallengeOutput, code: CodeEntry) -> bool:
    """
    Check that evidence_used is grounded in the code's real data.
    Since benchmarker now resolves indices → real text in pipeline,
    we verify the text appears in the code's actual evidence/criteria corpus.
    """
    cited = challenge.evidence_used.strip()
    if not cited:
        return False

    # Build real corpus
    corpus_parts = list(code.evidence_snippets)
    for c in code.inclusion + code.exclusion:
        corpus_parts.append(c.get("criterion", ""))
        corpus_parts.extend(c.get("examples", []))
    corpus = " ".join(corpus_parts).lower()

    # Each line of evidence_used should match something real
    # (lines are "[E-N] verbatim text" format from benchmarker)
    import re
    lines = [l.strip() for l in cited.split("\n") if l.strip()]
    for line in lines:
        # Strip the [E-N] prefix if present
        text = re.sub(r'^\[E\d+\]\s*', '', line)
        words = [w.lower() for w in text.split() if len(w) > 4]
        if not words:
            continue
        matches = sum(1 for w in words if w in corpus)
        if matches / len(words) >= 0.65:
            return True
    return False


def check_challenge(
    challenge: ChallengeOutput,
    component: ComponentType,
    model: str,
    base_url: str = "http://localhost:8000/v1",
    code: CodeEntry | None = None,
) -> FeasibilityOutput:
    # Pre-check: evidence_used must be grounded in real data (not hallucinated)
    if code is not None and not _evidence_grounded(challenge, code):
        logger.info("Gate: evidence not grounded in real data — dropping")
        return FeasibilityOutput(
            decision="invalid",
            reason="Evidence cited by benchmarker does not match any real source evidence or criteria text.",
        )

    is_conceptual = component in CONCEPTUAL_COMPONENTS
    gate_instructions = (
        GATE_INSTRUCTIONS_CONCEPTUAL if is_conceptual
        else GATE_INSTRUCTIONS_CRITERIA
    ).format(component=component)

    user = GATE_USER_TEMPLATE.format(
        component=component,
        challenge_question=challenge.challenge_question,
        critique_claim=challenge.critique_claim,
        failure_mode=challenge.failure_mode,
        failure_mechanism=challenge.failure_mechanism,
        evidence_used=challenge.evidence_used,
        gate_instructions=gate_instructions,
    )
    try:
        raw = call_llm(GATE_SYSTEM, user, model=model, base_url=base_url, temperature=0.0)
        data = parse_json_response(raw)
        decision = data.get("decision", "invalid").lower()
        if decision not in ("valid", "invalid"):
            decision = "invalid"
        # Hard override: no failure mechanism → always invalid
        if not data.get("has_failure_mechanism", True):
            decision = "invalid"
        reason = data.get("reason", "")
        # For criteria components only: evidence must function as counterexample
        if not is_conceptual and not data.get("evidence_used_as_counterexample", True):
            decision = "invalid"
            reason = "Evidence cited as background only, not as a counterexample."
        # Hard override: evidence text must semantically support the failure_mechanism
        if not is_conceptual and not data.get("evidence_supports_mechanism", True):
            decision = "invalid"
            reason = "Evidence text does not support the stated failure mechanism."
        return FeasibilityOutput(decision=decision, reason=reason)
    except Exception as exc:
        logger.warning("Gate check failed: %s", exc)
        return FeasibilityOutput(decision="invalid", reason=f"Gate error: {exc}")
