"""Greedy, sequence-aware duplicate filtering for visual search results with BLAS matrix prefiltering.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np


def filter_similar_results(
    results: List[Dict[str, Any]],
    embeddings: Dict[str, np.ndarray | List[float]],
    *,
    threshold: float = 0.85,
    id_key: str = "frame_id",
) -> List[Dict[str, Any]]:
    """Greedy suppression of near-duplicate visual search results based on cosine similarity.
    
    Uses BLAS matrix multiplication prefilter for fast $O(N)$ pair pruning.
    
    Args:
        results: Ranked list of search hit dictionaries.
        embeddings: Dict mapping result item ID to normalized float32 embedding vector.
        threshold: Cosine similarity cutoff above which later hits are suppressed (e.g., 0.85).
        id_key: Key inside each result dictionary referencing the embedding dict.

    Returns:
        Filtered list of search hit dictionaries with near-duplicates removed.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    # Normalize vectors
    normalized: Dict[str, np.ndarray] = {}
    for item_id, vec in embeddings.items():
        v = np.asarray(vec, dtype="float32").reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm > 1e-9:
            normalized[str(item_id)] = v / norm

    valid_indices = [
        i for i, res in enumerate(results)
        if str(res.get(id_key, "")) in normalized
    ]
    if not valid_indices:
        return list(results)

    # Fast matrix prefilter
    vectors = np.stack([normalized[str(results[i][id_key])] for i in valid_indices])
    similarities = vectors @ vectors.T
    guard = 4 * np.finfo(np.float32).eps * vectors.shape[1]
    
    positions = {res_idx: vec_idx for vec_idx, res_idx in enumerate(valid_indices)}
    kept: List[Dict[str, Any]] = []
    kept_positions: List[int] = []
    cutoff = threshold + 1e-6

    for i, res in enumerate(results):
        pos = positions.get(i)
        duplicate = False
        if pos is not None and kept_positions:
            candidates = np.asarray(kept_positions)
            # Fast BLAS pruning
            high_sim_candidates = candidates[similarities[pos, candidates] > cutoff - guard]
            duplicate = any(
                1.0 - (1.0 - float(np.dot(vectors[pos], vectors[prev]))) > cutoff
                for prev in high_sim_candidates
            )
        if not duplicate:
            kept.append(res)
            if pos is not None:
                kept_positions.append(pos)

    return kept
