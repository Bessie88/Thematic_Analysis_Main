"""Prompt templates for benchmarker, feasibility gate, answerer, and judge."""

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARKER / CHALLENGE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARKER_SYSTEM = """\
You are a qualitative research methods expert evaluating a thematic codebook.
Generate one specific, evidence-grounded challenge for a single component of a code.
Your critique must identify a concrete potential coding failure, not just a theoretical concern.
Only use the evidence provided; do not invent examples.
Cite evidence by index only."""

BENCHMARKER_INSTRUCTIONS_CRITERIA = """\
Challenge type: EVIDENCE-BASED

The challenge must:
1. Cite one or more evidence items by index only (e.g. [E3]). Do NOT copy or paraphrase evidence text.
2. Explain concretely how the cited evidence would be wrongly included, wrongly excluded, \
or left undecidable under the current {component} component.
3. Avoid purely theoretical concerns — show how a specific cited evidence item would \
actually be affected by the current {component}.

Respond in JSON:
{{
  "challenge_question": "<one concrete challenge question citing evidence by [E-N] index>",
  "critique_claim": "<specific critique hypothesis>",
  "failure_mode": "wrongly_excluded" or "wrongly_included" or "boundary_undecidable" or "unsupported_or_vague",
  "failure_mechanism": "<explain how the current {component} would cause the cited evidence to be miscoded or undecidable>",
  "evidence_indices": "<comma-separated index numbers, e.g. '1,3,5'>"
}}"""

BENCHMARKER_INSTRUCTIONS_CONCEPTUAL = """\
Challenge type: CONCEPTUAL

The challenge must identify a structural or conceptual flaw in the {component} component. Focus on:
- Is it ambiguous enough that different coders would interpret it differently?
- Does it overlap significantly with neighboring codes in ways that would cause confusion?
- Is it circular, operationally unusable, or misaligned with the label/definition?

You MAY cite evidence by index to illustrate the problem, but the core argument must stand \
on the conceptual flaw itself, not on whether a specific item would be miscoded.

Respond in JSON:
{{
  "challenge_question": "<one concrete challenge question about the {component}>",
  "critique_claim": "<specific critique of the conceptual flaw>",
  "failure_mode": "boundary_undecidable" or "unsupported_or_vague",
  "failure_mechanism": "<explain the specific flaw: why would different coders disagree, \
or what makes this {component} operationally unclear or misleading>",
  "evidence_indices": "<comma-separated indices if you cite evidence, or empty string>"
}}"""

BENCHMARKER_USER_TEMPLATE = """\
Codebook entry:

LABEL: {label}
DEFINITION: {definition}
INCLUSION CRITERIA:
{inclusion_text}
EXCLUSION CRITERIA:
{exclusion_text}

SOURCE EVIDENCE:
{evidence_text}

---
Component to evaluate: {component}
Scope: {scope}

{challenge_instructions}"""

COMPONENT_SCOPES = {
    "label": (
        "Identify specific ways this label could mislead a coder or overlap with a neighboring code. "
        "Focus on ambiguity in scope, misleading connotations, or failure to capture the core concept."
    ),
    "definition": (
        "Find operational gaps in this definition: where would two coders reach different conclusions? "
        "Look for circular reasoning, undefined key terms, or cases the definition leaves unresolved."
    ),
    "inclusion": (
        "Find evidence items that sit in gray zones the inclusion criteria do not clearly resolve. "
        "Look for patterns in the evidence that are omitted or handled inconsistently."
    ),
    "exclusion": (
        "Find evidence items that sit on the boundary between this code and a neighboring one. "
        "Look for cases where the exclusion criteria are too vague to give a coder a clear decision."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# FEASIBILITY GATE
# ─────────────────────────────────────────────────────────────────────────────

GATE_SYSTEM = """\
You check whether a generated codebook challenge is valid and usable.
You are not evaluating the codebook itself — only the quality of the challenge.
A valid challenge must demonstrate a concrete potential coding failure, \
not just a plausible theoretical concern."""

GATE_INSTRUCTIONS_CRITERIA = """\
This is an EVIDENCE-BASED challenge. A valid challenge must:
1. Address the "{component}" component specifically.
2. Cite actual evidence by index rather than invented content.
3. Use evidence as a counterexample or concrete failure case, not merely background.
4. Explain how the current {component} would cause a wrong inclusion, wrong exclusion, \
or undecidable boundary on a specific cited item.
5. The resolved evidence text must semantically support the stated failure_mechanism — \
being topic-related is not enough; the evidence must actually demonstrate the specific \
coding problem described.

Invalid if:
- It only says criteria "may miss" something without showing how cited evidence would be miscoded.
- Evidence cited illustrates the topic but does not test the current {component}.
- The evidence text does not actually support the failure_mechanism as described \
(e.g. the mechanism claims wrongly_excluded but the evidence clearly fits the inclusion criteria).

Respond in JSON:
{{
  "decision": "valid" or "invalid",
  "has_failure_mechanism": true or false,
  "evidence_used_as_counterexample": true or false,
  "evidence_supports_mechanism": true or false,
  "reason": "<one sentence>"
}}"""

GATE_INSTRUCTIONS_CONCEPTUAL = """\
This is a CONCEPTUAL challenge about the {component} itself. A valid challenge must:
1. Address the "{component}" component specifically.
2. Identify a concrete structural or operational flaw (ambiguity, circularity, overlap with \
neighboring codes, misalignment between label and definition).
3. Explain specifically why different coders would reach different decisions, or why the \
{component} is operationally unusable as written.

Invalid if:
- It only makes a vague claim that the {component} "could be clearer."
- It requires comparing against a gold standard.
- The flaw described is purely hypothetical with no grounding in the codebook text.

Respond in JSON:
{{
  "decision": "valid" or "invalid",
  "has_failure_mechanism": true or false,
  "evidence_used_as_counterexample": false,
  "reason": "<one sentence>"
}}"""

GATE_USER_TEMPLATE = """\
A challenge was generated for the "{component}" component of a codebook entry:

CHALLENGE QUESTION: {challenge_question}
CRITIQUE CLAIM: {critique_claim}
FAILURE MODE: {failure_mode}
FAILURE MECHANISM: {failure_mechanism}
EVIDENCE USED: {evidence_used}

Is this challenge valid?

{gate_instructions}"""


# ─────────────────────────────────────────────────────────────────────────────
# ANSWERER — codebook author defends the challenged component
# ─────────────────────────────────────────────────────────────────────────────

ANSWERER_SYSTEM = """\
You are the author of a qualitative thematic codebook defending your design choices.
A reviewer has challenged one component of one of your codes.
You must explicitly state how the cited evidence is handled by the current codebook criteria.
A general design rationale is not sufficient — you must map evidence to a specific criterion."""

ANSWERER_USER_TEMPLATE = """\
Your codebook entry:

LABEL [LABEL]: {label}
DEFINITION [DEF]: {definition}
INCLUSION CRITERIA:
{inclusion_text}
EXCLUSION CRITERIA:
{exclusion_text}

SOURCE EVIDENCE (cite by index only, e.g. [E3]):
{evidence_text}

Valid criterion IDs for this entry: {valid_criterion_ids}

---
Challenge on the "{component}" component:

CHALLENGE QUESTION: {challenge_question}
CRITIQUE CLAIM: {critique_claim}
FAILURE MODE: {failure_mode}
FAILURE MECHANISM: {failure_mechanism}
EVIDENCE CITED BY BENCHMARKER: {evidence_used}

---
You must directly address how the cited evidence is handled by the current codebook.

Required:
1. State the disposition of the cited evidence: "included", "excluded", or "undecidable".
2. Identify which criterion determines this by its ID from the list above \
(e.g. "INC-2", "EXC-1", "LABEL", "DEF").
3. Explain how the cited evidence maps to that criterion.

Rules (strictly enforced):
- supporting_criterion_id MUST be one of the valid IDs listed above. \
Do NOT invent IDs or use free text.
- Evidence items: cite by index only (e.g. [E2]). Do NOT copy or paraphrase evidence text.
- Do NOT give only a general rationale such as "this was intentional."

Respond in JSON:
{{
  "evidence_disposition": "included" or "excluded" or "undecidable",
  "supporting_criterion_id": "<one ID from the valid criterion IDs list, or null if none applies>",
  "evidence_to_criteria_mapping": "<explain how the cited evidence maps to the criterion identified above>",
  "response": "<2-4 sentence defense>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# JUDGE / ADJUDICATOR
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You adjudicate a challenge against one component of a qualitative codebook.
Follow the three-step checklist exactly. Do not add considerations outside it."""

JUDGE_USER_TEMPLATE = """\
Codebook entry:

LABEL: {label}
DEFINITION: {definition}
INCLUSION CRITERIA:
{inclusion_text}
EXCLUSION CRITERIA:
{exclusion_text}

SOURCE EVIDENCE (ground truth):
{evidence_text}

---
Component evaluated: {component}

BENCHMARKER'S CHALLENGE: {challenge_question}
BENCHMARKER'S CRITIQUE: {critique_claim}
FAILURE MODE: {failure_mode}
FAILURE MECHANISM: {failure_mechanism}
EVIDENCE CITED: {evidence_used}

AUTHOR'S STRUCTURED DEFENSE:
- Evidence disposition: {evidence_disposition}
- Supporting criterion ID: {criterion_location} / {supporting_criterion}
- Evidence-to-criteria mapping: {evidence_to_criteria_mapping}
- Narrative: {answerer_response}

---
Adjudicate in this order:

STEP 1 — Classify evidence_role:
  background_only      : Evidence is cited as context; not shown to cause a wrong inclusion,
                         wrong exclusion, or undecidable boundary under the current component.
  plausible_concern    : Benchmarker identifies a possible limitation but does not show that
                         a specific cited evidence item would actually be miscoded.
  demonstrated_failure : Benchmarker shows concretely how a specific cited evidence item
                         would be wrongly included, wrongly excluded, or left undecidable.

STEP 2 — Classify author_resolution:
  If evidence_role is background_only or plausible_concern, set author_resolution = "not_resolved";
  it will not affect the decision.

  If evidence_role is demonstrated_failure:
    resolved             : The cited criterion genuinely determines the evidence disposition,
                           and the mapping shows concretely why this evidence falls under it.
    partially_resolved   : Criterion is plausibly relevant but mapping is weak or indirect.
    not_resolved         : Criterion does not support the disposition, or mapping is circular,
                           absent, or clearly wrong. A valid criterion ID alone is not enough.

STEP 3 — Apply decision:
  background_only                           → rejected
  plausible_concern                         → rejected
  demonstrated_failure + resolved           → rejected
  demonstrated_failure + partially_resolved → unclear
  demonstrated_failure + not_resolved       → upheld

Set material_failure = true only if decision = upheld; otherwise false.

Respond in JSON:
{{
  "evidence_role": "background_only" or "plausible_concern" or "demonstrated_failure",
  "author_resolution": "resolved" or "partially_resolved" or "not_resolved",
  "failure_type": "wrongly_excluded" or "wrongly_included" or "boundary_undecidable" or "none",
  "material_failure": true or false,
  "decision": "upheld" or "rejected" or "unclear",
  "reasoning": "<one to two sentences citing which step determined the outcome>"
}}"""
