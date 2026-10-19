"""Reciprocal Rank Fusion (RRF) module.
Combines ranked result lists from multiple retrieval sources/models (e.g., CLIP and SigLIP).
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional


def rrf(
    rankings: List[List[Dict[str, Any]]],
    *,
    key: str = "id",
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion over multiple ranking lists.
    
    Formula:
        RRF_score(d) = sum_{m in models} weight_m / (k + rank_m(d))
        
    Args:
        rankings: List of ranked result lists from different models/queries.
        key: Unique identifier key in the result dicts (e.g., 'id', 'video_id', 'filename').
        k: Smoothing constant (standard: 60).
        weights: Optional weights for each ranking list. Defaults to equal weights (1.0).

    Returns:
        List of merged hits sorted by rrf_score descending, with 'rrf_score', 'confidence',
        and 'evidence' attached.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings length")
    if any(float(weight) < 0 for weight in weights):
        raise ValueError("weights must not be negative")

    merged: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, float] = {}
    evidence: Dict[str, List[Dict[str, Any]]] = {}

    for ranking, weight in zip(rankings, weights):
        if float(weight) == 0:
            continue
        seen = set()
        for rank, hit in enumerate(ranking, start=1):
            identity = hit.get(key)
            if identity is None or identity in seen:
                continue
            seen.add(identity)
            merged.setdefault(identity, dict(hit))
            scores[identity] = scores.get(identity, 0.0) + float(weight) / (k + rank)
            evidence.setdefault(identity, []).append(dict(hit))

    if not scores:
        return []

    best = max(scores.values())
    output = []
    for identity, hit in merged.items():
        res = dict(hit)
        res["rrf_score"] = scores[identity]
        res["confidence"] = scores[identity] / best if best > 0 else 0.0
        res["evidence"] = evidence[identity]
        output.append(res)

    return sorted(output, key=lambda row: row["rrf_score"], reverse=True)
