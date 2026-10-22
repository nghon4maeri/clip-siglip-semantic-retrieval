"""Reciprocal Rank Fusion (RRF) and Score Normalization Strategies.

Adapted from production multimodal retrieval pipelines:
- RRF combines multiple ranked lists without requiring score scale calibration.
- Supports model weighting (e.g. SigLIP 768D vs CLIP 512D).
- Normalizes confidence relative to the top fused candidate.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[List[Dict[str, Any]]],
    *,
    key: str = "id",
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Merges multiple ranked candidate lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = sum_{m} w_m / (k + rank_m(d))

    Args:
        rankings: List of ranked lists, where each list contains dict items.
        key: Unique identifier key in the item dicts (e.g. 'id', 'frame_id', 'filename').
        k: Smoothing constant (default 60, optimal for IR/multimodal retrieval).
        weights: Optional weighting factor per ranking list.

    Returns:
        Sorted list of fused dicts with 'rrf_score', 'confidence', 'evidence', and 'rank'.
    """
    if k < 1:
        raise ValueError("RRF constant k must be >= 1")

    if not rankings:
        return []

    if weights is None:
        weights = [1.0] * len(rankings)

    if len(weights) != len(rankings):
        raise ValueError(f"Number of weights ({len(weights)}) must match rankings ({len(rankings)})")

    merged: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, float] = {}
    evidence: Dict[str, List[Dict[str, Any]]] = {}

    for ranking, weight in zip(rankings, weights):
        w = float(weight)
        if w <= 0:
            continue
        seen_in_ranking = set()
        for rank_idx, hit in enumerate(ranking, start=1):
            identity = hit.get(key)
            if identity is None or identity in seen_in_ranking:
                continue
            seen_in_ranking.add(identity)

            str_id = str(identity)
            if str_id not in merged:
                merged[str_id] = dict(hit)
                scores[str_id] = 0.0
                evidence[str_id] = []

            scores[str_id] += w / (k + rank_idx)
            evidence[str_id].append(dict(hit))

    if not scores:
        return []

    best_score = max(scores.values())
    output: List[Dict[str, Any]] = []

    for identity, hit in merged.items():
        raw_score = scores[identity]
        hit_copy = dict(hit)
        hit_copy["rrf_score"] = round(raw_score, 6)
        hit_copy["confidence"] = round(raw_score / (best_score or 1.0), 4)
        hit_copy["evidence"] = evidence[identity]
        # Keep original score if present or assign normalized confidence
        hit_copy["fused_score"] = hit_copy["confidence"]
        output.append(hit_copy)

    output.sort(key=lambda x: x["rrf_score"], reverse=True)
    for idx, item in enumerate(output, start=1):
        item["rank"] = idx

    return output


def linear_score_fusion(
    rankings: Sequence[List[Dict[str, Any]]],
    *,
    key: str = "id",
    weights: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Min-max normalized weighted linear score combination."""
    if not rankings:
        return []

    if weights is None:
        weights = [1.0] * len(rankings)

    normalized_rankings = []
    for ranking in rankings:
        if not ranking:
            normalized_rankings.append([])
            continue
        scores = [float(h.get("score", 0.0)) for h in ranking]
        s_min, s_max = min(scores), max(scores)
        denom = (s_max - s_min) if (s_max - s_min) > 1e-7 else 1.0
        norm_list = []
        for h in ranking:
            h_copy = dict(h)
            h_copy["_norm_score"] = (float(h.get("score", 0.0)) - s_min) / denom
            norm_list.append(h_copy)
        normalized_rankings.append(norm_list)

    merged: Dict[str, Dict[str, Any]] = {}
    fused_scores: Dict[str, float] = {}

    for norm_ranking, w in zip(normalized_rankings, weights):
        for h in norm_ranking:
            identity = str(h.get(key, ""))
            if not identity:
                continue
            if identity not in merged:
                merged[identity] = dict(h)
                fused_scores[identity] = 0.0
            fused_scores[identity] += float(w) * h["_norm_score"]

    output = []
    for identity, h in merged.items():
        h_copy = dict(h)
        h_copy.pop("_norm_score", None)
        h_copy["score"] = fused_scores[identity]
        output.append(h_copy)

    output.sort(key=lambda x: x["score"], reverse=True)
    for idx, item in enumerate(output, start=1):
        item["rank"] = idx

    return output
