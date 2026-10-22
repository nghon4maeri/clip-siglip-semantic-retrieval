"""Multi-Detail Aspect Search Strategy (Single-Frame Decomposition).

Adapted from the reference repository's DuyMultiDetailSearch:
- Allows one complex frame to be described across several distinct detail prompts
  (e.g., 'a red Ferrari sports car', 'rainy night in Tokyo', 'neon billboard reflections').
- Bypasses text encoder token length limitations (~64-77 tokens) by searching each
  facet independently and fusing via Reciprocal Rank Fusion (RRF).
- Frames ranking well across multiple details receive boosted confidence scores.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence
from .fusion import reciprocal_rank_fusion
from .similarity_filter import filter_near_duplicate_frames


def search_multi_detail(
    detail_queries: Sequence[str],
    retriever_fn: Callable[[str, int], List[Dict[str, Any]]],
    *,
    weights: Optional[Sequence[float]] = None,
    top_k: int = 20,
    per_detail_candidates: int = 100,
    dedup_threshold: Optional[float] = 0.90,
    key: str = "id",
) -> List[Dict[str, Any]]:
    """Executes multi-detail aspect retrieval and fuses candidates using RRF.

    Args:
        detail_queries: List of detail/facet query strings.
        retriever_fn: Callable taking (query_string, k) and returning a list of hit dicts.
        weights: Optional importance weights per detail query.
        top_k: Number of final results to return.
        per_detail_candidates: Headroom candidates per query (default 100).
        dedup_threshold: Optional cosine near-duplicate cutoff.
        key: Identity key.

    Returns:
        Fused ranked list of candidate frames.
    """
    clean_queries = [q.strip() for q in detail_queries if q.strip()]
    if not clean_queries:
        return []

    if len(clean_queries) == 1:
        single_hits = retriever_fn(clean_queries[0], top_k)
        if dedup_threshold is not None:
            single_hits = filter_near_duplicate_frames(single_hits, threshold=dedup_threshold, key=key)
        return single_hits[:top_k]

    # Retrieve candidates for each detail independently
    rankings: List[List[Dict[str, Any]]] = []
    for q in clean_queries:
        hits = retriever_fn(q, per_detail_candidates)
        rankings.append(hits)

    # Fuse with RRF
    fused = reciprocal_rank_fusion(rankings, key=key, weights=weights, k=60)

    # Apply near-duplicate suppression if requested
    if dedup_threshold is not None:
        fused = filter_near_duplicate_frames(fused, threshold=dedup_threshold, key=key)

    return fused[:top_k]
