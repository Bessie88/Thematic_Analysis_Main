"""Schema helpers for enriched codebook review payloads (v1 / v2)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

SourceKind = Literal["llm", "human", "llm+edited"]
ClusterStatus = Literal["keep", "drop"]
OperationType = Literal["merge", "split", "drop"]


class ClusterEntry(TypedDict, total=False):
    label: str
    description: str
    confidence: int
    source: SourceKind
    needs_more_evidence: bool
    status: ClusterStatus


class MergeOperation(TypedDict, total=False):
    type: Literal["merge"]
    from_cluster_ids: List[str]
    target_cluster_id: str
    label: str


class SplitPart(TypedDict, total=False):
    new_cluster_id: str
    label: str
    code_ids: List[str]


class SplitOperation(TypedDict, total=False):
    type: Literal["split"]
    from_cluster_id: str
    splits: List[SplitPart]


class DropOperation(TypedDict, total=False):
    type: Literal["drop"]
    cluster_id: str
    code_disposition: Literal["delete", "orphan"]


class EnrichedCodebook(TypedDict, total=False):
    version: int
    clusters: Dict[str, ClusterEntry]
    operations: List[Dict[str, Any]]
    cluster_to_codes: Dict[str, List[str]]


def default_cluster_entry(
    label: str,
    *,
    description: str = "",
    confidence: int = 1,
    source: SourceKind = "llm",
) -> ClusterEntry:
    return {
        "label": label,
        "description": description,
        "confidence": confidence,
        "source": source,
        "needs_more_evidence": False,
        "status": "keep",
    }


def build_enriched_from_artifacts(
    codebook: Dict[str, str],
    cluster_to_codes: Dict[str, List[str]],
    confidence: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    version: int = 1,
) -> EnrichedCodebook:
    """Build enriched v1 payload from flat pipeline artifacts."""
    confidence = confidence or {}
    clusters: Dict[str, ClusterEntry] = {}
    for cid, label in codebook.items():
        conf = confidence.get(str(cid), confidence.get(cid, {}))
        if not isinstance(conf, dict):
            conf = {}
        clusters[str(cid)] = default_cluster_entry(
            label,
            description=str(conf.get("rationale") or ""),
            confidence=int(conf.get("confidence", 1) or 1),
            source="llm",
        )
    return {
        "version": version,
        "clusters": clusters,
        "operations": [],
        "cluster_to_codes": {str(k): list(v) for k, v in cluster_to_codes.items()},
    }


def flat_codebook_from_enriched(enriched: EnrichedCodebook) -> Dict[str, str]:
    """Extract flat label map from enriched payload."""
    clusters = enriched.get("clusters") or {}
    return {str(cid): str(entry.get("label") or f"Cluster {cid}") for cid, entry in clusters.items()}
