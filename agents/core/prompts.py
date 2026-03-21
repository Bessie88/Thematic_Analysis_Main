"""LLM prompt strings for the GT pipeline. Tools import these to keep prompt iteration separate from logic."""
from typing import Optional


def open_coding_prompt(
    research_question: str,
    text: str,
    validator_feedback: Optional[str] = None,
) -> str:
    """Build the open-coding prompt. If validator_feedback is given, include the reviewer feedback block."""
    feedback_section = ""
    if validator_feedback:
        feedback_section = f"""
A reviewer found issues with the previous codes. Use this feedback to improve:
{validator_feedback}

Revise your codes accordingly. Output the same format as below.
"""
    return f"""
You are doing Open Coding for thematic analysis on ONE user review.

Research Question: {research_question}
Focus on aspects of the review that are relevant to the research question above.

Rules:
- Produce 1 to 5 codes total (depending on content and length of the review).
- Each code must be a short noun phrase (2–6 words).
- Codes must be distinct (no near-duplicates).
- Each code must name a specific aspect AND its quality or direction — not just a neutral topic.
  Good: "laggy multiplayer matchmaking", "intuitive inventory controls", "frustrating difficulty spike"
  Bad: "matchmaking", "controls", "difficulty"
- Codes must be grounded in the review text — do not invent concepts not present.
- For short reviews under 50 words, produce at most 2 codes. Only add a third if the review clearly contains two distinct experiential concepts.
{feedback_section}

Output exactly as bullet points:
- Code: <code>
  Evidence: "<short quote from the review>"
  Note: <one short phrase why this code fits>

Review:
{text}
"""


def validate_open_codes_prompt(
    research_question: str,
    text: str,
    generated_codes: str,
) -> str:
    """Return the reviewer prompt with research question, input text, generated codes, and PASS/FAIL instructions."""
    return f"""You are reviewing qualitative codes generated from user feedback for a research question.

Research Question: {research_question}

Input text (the review):
{text}

Generated codes:
{generated_codes}

Check:
1. Codes are grounded in the data (evidence in the review supports each code).
2. Codes are not duplicates or near-duplicates of each other.
3. Codes are concise concepts (short noun phrases, not vague or hallucinated).
4. Codes are evaluatively specific — each names a concrete aspect AND its quality or
   direction (e.g. "poor enemy AI behaviour", not just "enemy AI").
   A code that is a neutral topic label with no direction should be flagged as FAIL.

Respond with exactly one of:
- PASS
or
- FAIL
Issues:
- <first issue>
- <second issue>
...

If PASS, you may add a single line of explanation after PASS. If FAIL, list specific issues so the coder can revise."""


def high_level_code_generation_prompt(bulleted: str, research_question: str) -> str:
    """Return the prompt that asks for JSON with label, confidence, rationale for one cluster."""
    rq_line = ""
    if research_question:
        rq_line = f"\nResearch Question: {research_question}\nGenerate with the research question in mind.\n"
    return f"""The following open codes belong to one cluster:

{bulleted}
{rq_line}
Output a single JSON object with:
- "label": one short high-level label (2-6 words) for this cluster
- "confidence": integer 1-5 (5 = very coherent, 1 = low coherence)
- "rationale": one sentence explaining why the cluster coheres or doesn't

Output ONLY valid JSON, no other text. Example: {{"label": "Interface usability issues", "confidence": 4, "rationale": "Codes all relate to UI and usability."}}"""


def refine_cluster_assignments_prompt(label: str, bulleted: str, other_str: str) -> str:
    """Return the prompt for identifying codes that belong in another cluster (MOVE or NONE).
    other_str contains only the top-5 most similar clusters by embedding distance.
    """
    return f"""You are reviewing the codes assigned to a cluster labeled "{label}".

Codes in this cluster:
{bulleted}

The only permitted move targets are these 5 clusters (chosen because they are the most similar to "{label}"):
{other_str}

A code should be moved ONLY if ALL of the following are true:
1. It shares zero conceptual overlap with "{label}" — not just a weaker fit, but genuinely no overlap.
2. It maps unambiguously to exactly one of the five clusters above — not a toss-up between two.
3. You would bet confidently on this move; any doubt means leave it.

For each code that meets all three criteria, output exactly:
MOVE: "{{code}}" → "{{target cluster label}}"

If no codes meet all three criteria, output: NONE

Do not move codes that are borderline, tangentially related, or where you are choosing the "least bad" option from the list. When in doubt: NONE."""

def relationship_classification_prompt(
    node_a: str,
    node_b: str,
    research_question: str,
) -> str:
    """Single prompt for hierarchy_construction and cross-cluster linking: classify A/B as equivalent/subsumes/subsumed_by/orthogonal."""
    return f"""Given two codes from a thematic analysis:
A: "{node_a}"
B: "{node_b}"

Research Question: {research_question}

Classify the relationship between A and B as exactly one of:
- "equivalent": A and B mean essentially the same thing (should be merged)
- "subsumes": A is more general/abstract than B (A contains B)
- "subsumed_by": A is more specific than B (B contains A)
- "orthogonal": A and B are distinct concepts with no hierarchical relationship

Output ONLY valid JSON: {{"relation": "<one of the four>", "reason": "<brief explanation>"}}"""


def research_report_prompt(research_question: str, graph_text: str) -> str:
    """Prompt for final qualitative synthesis from the global thematic graph (markdown output)."""
    return f"""You are a qualitative researcher writing a short synthesis for a grounded-theory style analysis.

Research question:
{research_question}

You are given a **thematic graph** as text: a list of **nodes** (theme codes) and **edges** (directed parent → child relationships interpreted from the analysis pipeline). Use ONLY information supported by these nodes and edges. Do not invent themes or relations that are not reflected in the graph text.

Thematic graph:
{graph_text}

Write your response in **markdown** with exactly these sections:

## Research question
Restate the research question in one sentence (you may quote it).

## Graph structure
One short line: how many nodes and how many edges were provided (or stated in the graph header).

## Research answer
A concise answer to the research question in **a few sentences** (about 3–6), grounded strictly in the nodes and edges above. If the graph is ambiguous or thin for the question, say so briefly and still summarize what the graph does support.

Do not output JSON. Do not add long bullet lists unless essential."""
