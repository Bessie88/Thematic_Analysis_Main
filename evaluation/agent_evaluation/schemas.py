"""Data schemas for the CRB-style codebook evaluation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


COMPONENTS = ("label", "definition", "inclusion", "exclusion")
ComponentType = Literal["label", "definition", "inclusion", "exclusion"]

JudgeDecision = Literal["upheld", "rejected", "unclear"]
FeasibilityDecision = Literal["valid", "invalid"]


# ── Input structures ──────────────────────────────────────────────────────────

@dataclass
class CodeEntry:
    code_id: str
    label: str
    definition: str
    inclusion: list[dict]   # list of {criterion, code_ids, examples}
    exclusion: list[dict]
    level: str = "cluster"          # "cluster" | "meta_theme"
    evidence_snippets: list[str] = field(default_factory=list)  # from open-codes .md
    neighbor_evidence_snippets: list[str] = field(default_factory=list)  # from other clusters (for exclusion)

    def component_text(self, component: ComponentType) -> str:
        if component == "label":
            return self.label
        if component == "definition":
            return self.definition
        if component == "inclusion":
            parts = [c["criterion"] for c in self.inclusion]
            return " | ".join(parts)
        if component == "exclusion":
            parts = [c["criterion"] for c in self.exclusion]
            return " | ".join(parts)
        raise ValueError(f"Unknown component: {component}")

    def all_examples(self) -> list[str]:
        examples = []
        for c in self.inclusion + self.exclusion:
            examples.extend(c.get("examples", []))
        examples.extend(self.evidence_snippets)
        return list(dict.fromkeys(examples))  # deduplicate, preserve order


# ── Challenge generation output ───────────────────────────────────────────────

@dataclass
class ChallengeOutput:
    challenge_question: str
    critique_claim: str
    evidence_used: str
    failure_mode: str = "unsupported_or_vague"
    failure_mechanism: str = ""


# ── Feasibility gate output ───────────────────────────────────────────────────

@dataclass
class FeasibilityOutput:
    decision: FeasibilityDecision
    reason: str


# ── Answerer structured output ────────────────────────────────────────────────

@dataclass
class AnswererOutput:
    response: str                      # 2-4 sentence narrative defense
    evidence_disposition: str          # "included" | "excluded" | "undecidable" | ""
    supporting_criterion: str          # resolved criterion text (looked up from codebook)
    supporting_criterion_id: str       # ID cited by model, e.g. "INC-2" (empty if invalid)
    criterion_location: str            # "label"|"definition"|"inclusion"|"exclusion"|"none"
    evidence_to_criteria_mapping: str  # how cited evidence maps to the criterion


# ── Judge output ──────────────────────────────────────────────────────────────

@dataclass
class JudgeOutput:
    decision: JudgeDecision
    reasoning: str
    evidence_role: str = ""
    failure_type: str = ""
    author_resolution: str = ""        # "resolved"|"partially_resolved"|"not_resolved"
    material_failure: bool = False


# ── Final evaluation item (one row in output table) ──────────────────────────

@dataclass
class EvalItem:
    generator_model: str
    answerer_model: str
    benchmarker_model: str
    codebook_id: str
    code_id: str
    level: str                         # "cluster" | "meta_theme"
    component: ComponentType
    challenge_question: str
    critique_claim: str
    failure_mode: str
    failure_mechanism: str
    evidence_used: str
    feasibility_decision: str          # valid / invalid / dropped
    answerer_response: Optional[str]              # narrative defense
    answerer_evidence_disposition: Optional[str]  # included / excluded / undecidable
    answerer_supporting_criterion_id: Optional[str]  # e.g. "INC-2" (empty if invalid/absent)
    answerer_supporting_criterion: Optional[str]  # resolved codebook text for that ID
    answerer_criterion_location: Optional[str]    # label/definition/inclusion/exclusion/none
    answerer_evidence_mapping: Optional[str]      # evidence→criterion mapping
    judge_decision: Optional[str]      # upheld / rejected / unclear / None
    judge_reasoning: Optional[str]
    judge_evidence_role: Optional[str]
    judge_failure_type: Optional[str]
    judge_author_resolution: Optional[str]        # resolved/partially_resolved/not_resolved
    judge_material_failure: Optional[bool]
    outcome_y: Optional[int]           # 1=pass, 0=fail, None=dropped
    drop_reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "generator_model":                self.generator_model,
            "answerer_model":                 self.answerer_model,
            "benchmarker_model":              self.benchmarker_model,
            "codebook_id":                    self.codebook_id,
            "code_id":                        self.code_id,
            "level":                          self.level,
            "component":                      self.component,
            "challenge_question":             self.challenge_question,
            "critique_claim":                 self.critique_claim,
            "failure_mode":                   getattr(self, "failure_mode", ""),
            "failure_mechanism":              getattr(self, "failure_mechanism", ""),
            "evidence_used":                  self.evidence_used,
            "feasibility_decision":           self.feasibility_decision,
            "answerer_response":                 self.answerer_response,
            "answerer_evidence_disposition":     self.answerer_evidence_disposition,
            "answerer_supporting_criterion_id":  self.answerer_supporting_criterion_id,
            "answerer_supporting_criterion":     self.answerer_supporting_criterion,
            "answerer_criterion_location":       self.answerer_criterion_location,
            "answerer_evidence_mapping":         self.answerer_evidence_mapping,
            "judge_decision":                 self.judge_decision,
            "judge_reasoning":               self.judge_reasoning,
            "judge_evidence_role":            self.judge_evidence_role,
            "judge_failure_type":             self.judge_failure_type,
            "judge_author_resolution":        self.judge_author_resolution,
            "judge_material_failure":         self.judge_material_failure,
            "outcome_y":                      self.outcome_y,
            "drop_reason":                    self.drop_reason,
        }
